// This file is derived from cache.cc in the LevelDB projec
//
//   Some portions copyright (c) 2011 The LevelDB Authors. All rights reserved.
//   Use of this source code is governed by a BSD-style license that can be
//   found in the LICENSE file.
//
// ------------------------------------------------------------
// This file implements a cache based on the NVML library (http://pmem.io),
// specifically its "libvmem" and "libpmemobj" components. This library makes
// it easy to program against persistent memory hardware by exposing an API which
// parallels malloc/free, but also provides easy load/store access to persistent
// memory.
//
// We use this API to implement a cache which treats persistent memory or
// non-volatile memory as if it were a larger cheaper bank of volatile memory.
// We currently make no use of its persistence properties.

// Currently, we only store key/value in NVM. All other data structures such as the
// ShardedLRUCache instances, hash table, etc are in DRAM. The assumption is that
// the ratio of data stored vs overhead is quite high.


#include "kudu/util/nvm_cache.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <libpmemobj.h>
#include <libvmem.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <libpmemobj.h>
#include <libvmem.h>

#include "kudu/gutil/atomicops.h"
#include "kudu/gutil/atomic_refcount.h"
#include "kudu/gutil/dynamic_annotations.h"
#include "kudu/gutil/gscoped_ptr.h"
#include "kudu/gutil/bits.h"
#include "kudu/gutil/hash/city.h"
#include "kudu/gutil/macros.h"
#include "kudu/gutil/port.h"
#include "kudu/gutil/ref_counted.h"
#include "kudu/gutil/stl_util.h"
#include "kudu/util/cache.h"
#include "kudu/util/cache_metrics.h"
#include "kudu/util/flag_tags.h"
#include "kudu/util/locks.h"
#include "kudu/util/metrics.h"
#include "kudu/util/nvm_cache.h"
#include "kudu/util/slice.h"
#include "kudu/util/path_util.h"

// These values should be changed if there is an incompatible change to the
// format of any of the persistent on-media layout. If there is the
// pool will not open. This is a temporary upgrade fix to ensure
// no pool corruption occurs. It depends on the developer to determine
// what constitues an incompatible change and if the pool can be
// migrated to the new format.
#define NVMCACHE_MAJOR_VERSION 1
#define NVMCACHE_MINOR_VERSION 0

DEFINE_string(nvm_cache_path, "/pmem",
              "The path at which the NVM cache will try to allocate its memory. "
              "This can be a tmpfs or ramfs for testing purposes. "
              "It must be an existing directory.");
TAG_FLAG(nvm_cache_path, experimental);

DEFINE_string(nvm_cache_pool, "pm_cache",
              "The pool name given for the NVM cache to allocate its memory "
              "when running in persistent mode. It is a file created within "
              "the directory specified by nvm_cache_path.  This can be a tmpfs "
              "or ramfs for testing purposes.");
TAG_FLAG(nvm_cache_pool, experimental);

DEFINE_bool(nvm_cache_persistent, true,
            "The nvm_cache_persistent flag indicates whether the data "
            "in the cache is persistent across process restart. The default "
            "behavior(true) is to persist data across process restart.");

DEFINE_int32(nvm_cache_allocation_retry_count, 10,
             "The number of times that the NVM cache will retry attempts to allocate "
             "memory for new entries. In between attempts, a cache entry will be "
             "evicted.");
TAG_FLAG(nvm_cache_allocation_retry_count, advanced);
TAG_FLAG(nvm_cache_allocation_retry_count, experimental);

DEFINE_bool(nvm_cache_simulate_allocation_failure, false,
            "If true, the NVM cache will inject failures in calls to vmem_malloc "
            "and pmem_malloc for testing.");
TAG_FLAG(nvm_cache_simulate_allocation_failure, unsafe);

DEFINE_int32(nvm_cache_major_version, NVMCACHE_MAJOR_VERSION,
            "The major version for the current cache. This value is only used "
            "for testing.");
TAG_FLAG(nvm_cache_major_version, hidden);
TAG_FLAG(nvm_cache_major_version, runtime);

DECLARE_bool(cache_force_single_shard);

namespace kudu {

namespace {

using std::shared_ptr;
using std::string;

// Used for memory allocation of cache objects
enum AllocType {
  DRAM_ALLOC,
  VMEM_ALLOC,
  PMEM_ALLOC
};


// This is a variable length structure. The length of the structure is
// determined by the key and value sizes. This structure is the physical entry
// that is persisted as the pmemobj object.
struct KeyVal {
  uint32_t  key_len;
  uint32_t  value_len;

  // This member is set at the very end prior to persisting the KeyVal
  // object. This means that in the case of an interruption in service
  // the pmemobj object is not considered complete.
  // If 'valid' is not set then upon restart this entry is discarded.
  uint8_t   valid;
  uint8_t   kv_data[]; // holds key and value data
};

// When creating a new pool an object is created to store the current major
// and minor versions for the Key/Value layout.
struct KeyValVersion {
  int cache_major;
  int cache_minor;
};

// These define the structures which make up the layout of the pmemobj persistent
// pool. All objects are of the same type, struct KeyVal, except for the root
// object. There is only 1 root object.
POBJ_LAYOUT_BEGIN(kudublockcache);

// This is a version object that allows the consumer of the cache to determine
// what the current version level of the cache is. It is expected that the
// consumer of the cache will check the version prior to operating on the data
// in the cache. There is only 1 object of this type in the pool.
POBJ_LAYOUT_ROOT(kudublockcache, struct KeyValVersion);
POBJ_LAYOUT_TOID(kudublockcache, struct KeyVal);
POBJ_LAYOUT_END(kudublockcache);

// Constructor for persistent mode key/value structure. The non-volatile
// constructor is guaranteed to be atomic. The valid member of the KeyVal
// structure is used to determine if an entry in the cache is valid upon
// restart. This valid bit is set to 1 at the final stage.
int KvConstructor(PMEMobjpool* pop, void* ptr, void* arg) {
  struct KeyVal* kv = static_cast<struct KeyVal*>(ptr);
  kv->valid = 0;
  pmemobj_persist(pop, &kv->valid, sizeof(kv->valid));
  return 0;
}

// Finds the persistent memory object id for the given ptr and offset.
// The 'offset' value is the distance between the structure member and the
// enclosing structure. 'ptr' is the address of member for the instance of
// the structure.
PMEMoid FindOid(const uint8_t* ptr, size_t offset, PMEMobjpool* pop,
                       PMEMoid root) {
  DCHECK_GT(reinterpret_cast<uintptr_t>(ptr), offset);
  // Return OID of key_val structure. It's possible there will not be a
  // valid oid. It is up to the caller to verify this.
  PMEMoid newoid = OID_NULL;
  uintptr_t kv_ptr = reinterpret_cast<uintptr_t>(ptr - offset);
  newoid.off = reinterpret_cast<uintptr_t>(kv_ptr) - reinterpret_cast<uintptr_t>(pop);
  newoid.pool_uuid_lo = root.pool_uuid_lo;
  return newoid;
}

using std::vector;

typedef simple_spinlock MutexType;

// LRU cache implementation

// An entry is a variable length heap-allocated structure when running in volatile
// mode.  When operating in persistent mode the structure is fixed length.
// The entries are kept in a circular doubly linked list ordered by access time.

// For persistent memory there are two use cases for allocation of the LRUHandle.
// 1. When running in volatile mode the LRUHandle is allocated from the volatile
//    persistent memory pool. It is managed as part of the pool. This is similar
//    behavior to the DRAM cache.
// 2. When running in persistent mode the LRUHandle is allocated from DRAM.
// In either case the LRUHandle is never persisted.
//
// Entries are kept in a circular doubly linked list ordered by access time.
struct LRUHandle {
  Cache::EvictionCallback* eviction_callback;
  LRUHandle* next_hash;
  LRUHandle* next;
  LRUHandle* prev;
  bool repopulated;
  size_t charge;      // TODO(opt): Only allow uint32_t?
  uint32_t key_length;
  uint32_t val_length;
  Atomic32 refs;
  uint32_t hash; // Hash of key(); used for fast sharding and comparisons
  uint8_t* kv_data; // Either pointer to pmem or space for volatile pmem.

  Slice key() const {
    return Slice(kv_data, key_length);
  }

  Slice value() const {
    return Slice(&kv_data[key_length], val_length);
  }

  uint8_t* val_ptr() {
    return &kv_data[key_length];
  }
};

// We provide our own simple hash table since it removes a whole bunch
// of porting hacks and is also faster than some of the built-in hash
// table implementations in some of the compiler/runtime combinations
// we have tested.  E.g., readrandom speeds up by ~5% over the g++
// 4.4.3's builtin hashtable.
class HandleTable {
 public:
  HandleTable() : length_(0), elems_(0), list_(NULL) { Resize(); }
  ~HandleTable() { delete[] list_; }

  LRUHandle* Lookup(const Slice& key, uint32_t hash) {
    return *FindPointer(key, hash);
  }

  LRUHandle* Insert(LRUHandle* h) {
    LRUHandle** ptr = FindPointer(h->key(), h->hash);
    LRUHandle* old = *ptr;
    h->next_hash = (old == NULL ? NULL : old->next_hash);
    *ptr = h;
    if (old == NULL) {
      ++elems_;
      if (elems_ > length_) {
        // Since each cache entry is fairly large, we aim for a small
        // average linked list length (<= 1).
        Resize();
      }
    }
    return old;
  }

  LRUHandle* Remove(const Slice& key, uint32_t hash) {
    LRUHandle** ptr = FindPointer(key, hash);
    LRUHandle* result = *ptr;
    if (result != NULL) {
      *ptr = result->next_hash;
      --elems_;
    }
    return result;
  }

 private:
  // The table consists of an array of buckets where each bucket is
  // a linked list of cache entries that hash into the bucket.
  uint32_t length_;
  uint32_t elems_;
  LRUHandle** list_;

  // Return a pointer to slot that points to a cache entry that
  // matches key/hash.  If there is no such cache entry, return a
  // pointer to the trailing slot in the corresponding linked list.
  LRUHandle** FindPointer(const Slice& key, uint32_t hash) {
    LRUHandle** ptr = &list_[hash & (length_ - 1)];
    while (*ptr != NULL &&
           ((*ptr)->hash != hash || key != (*ptr)->key())) {
      ptr = &(*ptr)->next_hash;
    }
    return ptr;
  }

  void Resize() {
    uint32_t new_length = 16;
    while (new_length < elems_ * 1.5) {
      new_length *= 2;
    }
    LRUHandle** new_list = new LRUHandle*[new_length];
    memset(new_list, 0, sizeof(new_list[0]) * new_length);
    uint32_t count = 0;
    for (uint32_t i = 0; i < length_; i++) {
      LRUHandle* h = list_[i];
      while (h != NULL) {
        LRUHandle* next = h->next_hash;
        uint32_t hash = h->hash;
        LRUHandle** ptr = &new_list[hash & (new_length - 1)];
        h->next_hash = *ptr;
        *ptr = h;
        h = next;
        count++;
      }
    }
    DCHECK_EQ(elems_, count);
    delete[] list_;
    list_ = new_list;
    length_ = new_length;
  }
};

// A single shard of sharded cache.
class NvmLRUCache {
 public:
  NvmLRUCache(PMEMobjpool* pop, PMEMoid root, VMEM* vmp);
  ~NvmLRUCache();

  // Separate from constructor so caller can easily make an array of LRUCache
  void SetCapacity(size_t capacity) { capacity_ = capacity; }

  void SetMetrics(CacheMetrics* metrics) { metrics_ = metrics; }

  Cache::Handle* Insert(LRUHandle* h, Cache::EvictionCallback* eviction_callback);

  // Like Cache::Lookup, but with an extra "hash" parameter.
  Cache::Handle* Lookup(const Slice& key, uint32_t hash, bool caching);
  void Release(Cache::Handle* handle);
  void Erase(const Slice& key, uint32_t hash);

  // This method allocates the 'size' memory from one of the following types(atype):
  // DRAM_ALLOC, standard DRAM allocation
  // VMEM_ALLOC, NVM volatile memory
  // PMEM_ALLOC, NVM persistent memory
  void* AllocateAndRetry(size_t size, unsigned int type_num, AllocType atype,
                         pmemobj_constr constructor);

  // Fill in fields of LRUHandle instance.
  void PopulateCacheHandle(LRUHandle* e,
                           Cache::EvictionCallback* eviction_callback);

 private:
  void NvmLRU_Remove(LRUHandle* e);
  void NvmLRU_Append(LRUHandle* e);

  // Just reduce the reference count by 1.
  // Return true if last reference
  bool Unref(LRUHandle* e);

  // Call the user's eviction callback if defined.
  // To delete the entry from the persistent media, when in persistent
  // mode set deep to true. If set to false it will only remove the LRUHandle.
  void FreeEntry(LRUHandle* e, bool deep);

  // Evict the LRU item in the cache, adding it to the linked list
  // pointed to by 'to_remove_head'.
  void EvictOldestUnlocked(LRUHandle** to_remove_head);

  // Free all of the entries in the linked list that has to_free_head
  // as its head. Remove entries from the cache as well. This function
  // is only called after we the call to EvictOldestUnlocked which has
  // removed them from the LRUHandle list.
  void FreeLRUEntries(LRUHandle* to_free_head);

  // Allocate memory based on AllocType. The 'type_num' parameter
  // is set to the persistent object type to allocate. This is defined
  // types specified with POOL_LAYOUT_TOID when defining the pool structure.
  // There can be multiple types within a pool.
  void* NvmMalloc(size_t size, unsigned int type_num, AllocType atype,
                  pmemobj_constr constructor);

  // Wrapper around persistent memory allocation. A constructor is
  // required since we are allocating persistent memory in persistent mode.
  // The constructor ensures that the initialization of this memory is atomic.
  void* PmemMalloc(size_t size, unsigned int type_num, pmemobj_constr constructor);

  // Wrapper around vmem allocation which injects failures based on a flag.
  void* VmemMalloc(size_t size);

  // Method for determining if the cache is running in persistent mode.
  bool IsPersistentMode() const {
    return pop_ != nullptr;
  }

  // Initialized before use.
  size_t capacity_;

  // mutex_ protects the following state.
  MutexType mutex_;
  size_t usage_;

  // Dummy head of LRU list.
  // lru.prev is newest entry, lru.next is oldest entry.
  LRUHandle lru_;

  HandleTable table_;

  CacheMetrics* metrics_;

  // NVM pool variables
  PMEMobjpool* pop_; // Root address of persistent mode object pool
  PMEMoid root_; // Root object in persistent object pool
  VMEM* vmp_; // Root address of volatile memory pool

};

// NVM Cache can be either volatile or persistent
NvmLRUCache::NvmLRUCache(PMEMobjpool* pop, PMEMoid root, VMEM* vmp)
   : usage_(0),
    metrics_(NULL),
    pop_(pop),
    root_(root),
    vmp_(vmp) {
  // Make empty circular linked list
  lru_.next = &lru_;
  lru_.prev = &lru_;

}

NvmLRUCache::~NvmLRUCache() {
  for (LRUHandle* e = lru_.next; e != &lru_; ) {
    LRUHandle* next = e->next;
    DCHECK_EQ(e->refs, 1);  // Error if caller has an unreleased handle
    // If we are in persistent mode we don't delete the persistent cache
    // entries when the cache is closed. We only free the LRUHandle itself.
    if (Unref(e)) {
      bool deep = !IsPersistentMode();
      FreeEntry(e, deep);
    }
    e = next;
  }
}

void* NvmLRUCache::NvmMalloc(size_t size, unsigned int type_num, AllocType atype,
   pmemobj_constr constructor) {
  if (atype == PMEM_ALLOC) {
    return PmemMalloc(size, type_num, constructor);
  }
  if (atype == VMEM_ALLOC) {
    return VmemMalloc(size);
  }
  if (atype == DRAM_ALLOC) {
    return malloc(size);
  }
    LOG(FATAL) << "Unknown allocation type";
  return nullptr;
}

void* NvmLRUCache::PmemMalloc(size_t size, unsigned int type_num,
                              pmemobj_constr constructor) {
  if (PREDICT_FALSE(FLAGS_nvm_cache_simulate_allocation_failure)) return nullptr;
  PMEMoid oid = OID_NULL;
  size_t total_size = sizeof(struct KeyVal) + size;

  int status = pmemobj_alloc(pop_, &oid, total_size,
                             type_num, constructor, static_cast<void*>(&total_size));
  if (status) {
    return nullptr;
  }
  struct KeyVal* tmp = static_cast<KeyVal*>(pmemobj_direct(oid));
  DCHECK(tmp != nullptr);
  return &tmp->kv_data;
}

void* NvmLRUCache::VmemMalloc(size_t size) {
  if (PREDICT_FALSE(FLAGS_nvm_cache_simulate_allocation_failure)) {
    return NULL;
  }
  return vmem_malloc(vmp_, size);
}

// If 'deep' is set it means we want to delete the pmem object.
// If 'deep' is false we still delete the LRUHandle because
// we have multiple handles pointing to the pmem memory.
// This only applies to persistent mode.
void NvmLRUCache::FreeEntry(LRUHandle* e, bool deep) {
  DCHECK_EQ(ANNOTATE_UNPROTECTED_READ(e->refs), 0);
  if (e->eviction_callback) {
    e->eviction_callback->EvictedEntry(e->key(), e->value());
  }

  if (PREDICT_TRUE(metrics_)) {
    metrics_->cache_usage->DecrementBy(e->charge);
    metrics_->evictions->Increment();
  }

  if (IsPersistentMode()) {
    if (deep && e->repopulated) {
      // We delete the pmem resident data when in persistent mode.
      PMEMoid oid = FindOid(const_cast<uint8_t*>(e->kv_data),
                          offsetof(struct KeyVal, kv_data),
                          pop_, root_);
      DCHECK(pmemobj_direct(oid) != nullptr);
      POBJ_FREE(&oid);
    }
    delete e;
  } else {
    vmem_free(vmp_, e);
  }
}

bool NvmLRUCache::Unref(LRUHandle* e) {
  DCHECK_GT(ANNOTATE_UNPROTECTED_READ(e->refs), 0);
  return !base::RefCountDec(&e->refs);
}

// Allocate nvm memory. Try until successful or FLAGS_nvm_cache_allocation_retry_count
// has been exceeded.
void* NvmLRUCache::AllocateAndRetry(size_t size, unsigned int type_num,
                                    AllocType atype,
                                    pmemobj_constr constructor) {
  void* tmp;

  // There may be times that an allocation fails. With NVM we have
  // a fixed size to allocate from. If we cannot allocate the size
  // that was asked for, we will remove entries from the cache and
  // retry up to the configured number of retries. If this fails, we
  // return NULL, which will cause the caller to not insert anything
  // into the cache.
  tmp = NvmMalloc(size, type_num, atype, constructor);
  LRUHandle* to_remove_head = NULL;
  if (tmp == NULL) {
    std::unique_lock<MutexType> l(mutex_);

    int retries_remaining = FLAGS_nvm_cache_allocation_retry_count;
    while (tmp == NULL && retries_remaining-- > 0 && lru_.next != &lru_) {
      // Evict from hash table. This function keeps track of the list of
      // entries we have evicted. Then free from cache. Repeat until
      // We can allocate memory or run out of retries.
      EvictOldestUnlocked(&to_remove_head);
      FreeLRUEntries(to_remove_head);
      to_remove_head = nullptr;
      l.unlock();
      tmp = NvmMalloc(size, type_num, atype, constructor);
      l.lock();
    }
  }
  return tmp;
}

// Fill in fields of LRUHandle instance. If repopulate is set it means
// we are recovering data from the cache, either in the event of an
// unexpected error or cache shutdown. When we repopulate the handle
// the reference to it is only 1 since we do not have a request for
// a copy of it from the CFileReader. When a lookup request is made
// the reference count on this handle will be 2.
void NvmLRUCache::PopulateCacheHandle(LRUHandle* e,
                                      Cache::EvictionCallback* eviction_callback) {

  LRUHandle* to_remove_head = nullptr;
  e->eviction_callback = eviction_callback;

  // If this entry was created from a persistent entry in the cache we only have
  // 1 ref to it at this time, not 2. A ref count of 2 indicates it's a brand new
  // entry inserted via a cfile read. Once a reference is attached to it it is
  // no longer in repopulate mode. We simply up the refcount.
  if (e->repopulated) {
    e->refs = 1;
  } else {
    e->refs = 2;
  }

  if (PREDICT_TRUE(metrics_)) {
    metrics_->cache_usage->IncrementBy(e->charge);
    metrics_->inserts->Increment();
  }

  {
    std::lock_guard<MutexType> l(mutex_);

    NvmLRU_Append(e);

    LRUHandle* old = table_.Insert(e);
    if (old != nullptr) {
      NvmLRU_Remove(old);
      if (Unref(old)) {
        old->next = to_remove_head;
        to_remove_head = old;
      }
    }
    while (usage_ > capacity_ && lru_.next != &lru_) {
      EvictOldestUnlocked(&to_remove_head);
    }
  }
  FreeLRUEntries(to_remove_head);
}

void NvmLRUCache::NvmLRU_Remove(LRUHandle* e) {
  e->next->prev = e->prev;
  e->prev->next = e->next;
  usage_ -= e->charge;
}

void NvmLRUCache::NvmLRU_Append(LRUHandle* e) {
  // Make "e" newest entry by inserting just before lru_
  e->next = &lru_;
  e->prev = lru_.prev;
  e->prev->next = e;
  e->next->prev = e;
  usage_ += e->charge;
}

Cache::Handle* NvmLRUCache::Lookup(const Slice& key, uint32_t hash, bool caching) {
 LRUHandle* e;
  {
    std::lock_guard<MutexType> l(mutex_);
    e = table_.Lookup(key, hash);
    if (e != nullptr) {
      // If an entry exists, remove the old entry from the cache
      // and re-add to the end of the linked list.
      base::RefCountInc(&e->refs);
      NvmLRU_Remove(e);
      NvmLRU_Append(e);
    }
  }
  // Do the metrics outside of the lock.
  if (metrics_) {
    metrics_->lookups->Increment();
    bool was_hit = (e != nullptr);
    if (was_hit) {
      if (caching) {
        metrics_->cache_hits_caching->Increment();
      } else {
        metrics_->cache_hits->Increment();
      }
    } else {
      if (caching) {
        metrics_->cache_misses_caching->Increment();
      } else {
        metrics_->cache_misses->Increment();
      }
    }
  }

  return reinterpret_cast<Cache::Handle*>(e);
}

// Release any resources acquired during Lookup(). As a result, free
// only the LRUHandle entry not the data on the NVM device.
// In the case that this entry was from a repopulate from the
// persistent cache we only have 1 initial reference to it.
// We don't want to delete an entry in this case from the Release,
// only from an explicit free.
void NvmLRUCache::Release(Cache::Handle* handle) {
  LRUHandle* e = reinterpret_cast<LRUHandle*>(handle);
  bool last_reference = Unref(e);
  if (last_reference) {
    FreeEntry(e, true);
  }
}

void NvmLRUCache::EvictOldestUnlocked(LRUHandle** to_remove_head) {
  LRUHandle* old = lru_.next;
  NvmLRU_Remove(old);
  table_.Remove(old->key(), old->hash);
  if (Unref(old)) {
    old->next = *to_remove_head;
    *to_remove_head = old;
  }
}

void NvmLRUCache::FreeLRUEntries(LRUHandle* to_free_head) {
  while (to_free_head != NULL) {
    LRUHandle* next = to_free_head->next;
    FreeEntry(to_free_head, true);
    to_free_head = next;
  }
}

Cache::Handle* NvmLRUCache::Insert(LRUHandle* e, Cache::EvictionCallback* eviction_callback) {

  if (IsPersistentMode() && !e->repopulated) {

    // At the time of insertion we know we have succeeded in allocating
    // the pmem space we need. So, there will be an persistent object
    // created for this memory address.
    struct KeyVal* kv =
      reinterpret_cast<struct KeyVal*>(e->kv_data - offsetof(KeyVal, kv_data));

    if (!kv->valid) {
      pmemobj_flush(pop_, &kv->kv_data, e->key_length + e->val_length);

      kv->key_len = e->key_length;
      kv->value_len = e->val_length;

      pmemobj_persist(pop_, &kv->key_len, sizeof(e->key_length) + 
        sizeof(e->val_length));

      // Set valid bit to indicate entry is complete.
      kv->valid = 1;
      pmemobj_persist(pop_, &kv->valid, sizeof(kv->valid));
    }
  }

  // Populate the cache handle.
  PopulateCacheHandle(e, eviction_callback);
  return reinterpret_cast<Cache::Handle*>(e);
}

void NvmLRUCache::Erase(const Slice& key, uint32_t hash) {
  LRUHandle* e;
  bool last_reference = false;
  {
    std::lock_guard<MutexType> l(mutex_);
    e = table_.Remove(key, hash);
    if (e != NULL) {
      NvmLRU_Remove(e);
      last_reference = Unref(e);
    }
  }
  // mutex not held here
  // last_reference will only be true if e != NULL
  if (last_reference) {
    FreeEntry(e, true);
  }

}

// Determine the number of bits of the hash that should be used to determine
// the cache shard. This, in turn, determines the number of shards.
int DetermineShardBits() {
  int bits = PREDICT_FALSE(FLAGS_cache_force_single_shard) ?
      0 : Bits::Log2Ceiling(base::NumCPUs());
  VLOG(1) << "Will use " << (1 << bits) << " shards for LRU cache.";
  return bits;
}

class ShardedLRUCache : public Cache {
 private:
  gscoped_ptr<CacheMetrics> metrics_;
  vector<NvmLRUCache*> shards_;
  MutexType id_mutex_;
  uint64_t last_id_;

  PMEMobjpool* pop_; // Root address of persistent mode object pool
  PMEMoid root_; // Root object in persistent object pool
  VMEM* vmp_; // Root address of volatile memory pool

 // Number of bits of hash used to determine the shard.
  const int shard_bits_;

  static inline uint32_t HashSlice(const Slice& s) {
    return util_hash::CityHash64(
      reinterpret_cast<const char *>(s.data()), s.size());
  }

  uint32_t Shard(uint32_t hash) {
    // Widen to uint64 before shifting, or else on a single CPU,
    // we would try to shift a uint32_t by 32 bits, which is undefined.
    return static_cast<uint64_t>(hash) >> (32 - shard_bits_);
  }

 public:
  // "pop_" represents the block cache type. If this cache was created
  // with persistence this value will be an address of the persistent
  // pool, otherwise NULL.
  bool IsPersistentMode() {
    return pop_ != NULL;
  }

  explicit ShardedLRUCache(size_t capacity, PMEMobjpool* pop, PMEMoid root, VMEM* vmp)
      : metrics_(NULL),
        last_id_(0),
        pop_(pop),
        root_(root),
        vmp_(vmp),
        shard_bits_(DetermineShardBits()) {

    int num_shards = 1 << shard_bits_;
    const size_t per_shard = (capacity + (num_shards - 1)) / num_shards;
    for (int s = 0; s < num_shards; s++) {
      // Create shard and insert at end of vector list.
      gscoped_ptr<NvmLRUCache> shard(new NvmLRUCache(pop_, root_, vmp_));
      shard->SetCapacity(per_shard);
      shards_.push_back(shard.release());
    }

    if (IsPersistentMode()) {
      TOID(struct KeyVal) kv;

      // Populate a shard wtih existing entries(if any). A nullptr value breaks
      // us out of the loop, and means that there are no entries.

      // Since there are multiple object types in the pool we use the FOREACH_TYPE
      // and filter only on the TOID(struct KeyVal).
      POBJ_FOREACH_TYPE(pop_, kv) {
        if (D_RO(kv) == nullptr) {
          // This will only happen if there are no entries in the pool.
          break;
        }
        if (D_RO(kv)->valid == 1) {
          LRUHandle* e = new LRUHandle;
          e->repopulated = true;
          e->kv_data = const_cast<uint8_t*>(D_RO(kv)->kv_data);
          e->key_length = D_RO(kv)->key_len;
          e->val_length = D_RO(kv)->value_len;
          e->hash = HashSlice(e->key());
          e->charge = sizeof(struct KeyVal) + D_RO(kv)->key_len + D_RO(kv)->value_len;
          Insert(reinterpret_cast<PendingHandle*>(e), nullptr);
        } else {
          POBJ_FREE(&kv);
        }
      }
    }
  }

 virtual ~ShardedLRUCache() {

   STLDeleteElements(&shards_);
    // We have both volatile and persistent cache. Only delete the cache if
    // we are in volatile mode.
    if (IsPersistentMode()) {
      pmemobj_close(pop_);
    } else {
      vmem_delete(vmp_);
    }
  }

  virtual Handle* Insert(PendingHandle* handle,
                         Cache::EvictionCallback* eviction_callback) OVERRIDE {
    LRUHandle* h = reinterpret_cast<LRUHandle*>(DCHECK_NOTNULL(handle));
    return shards_[Shard(h->hash)]->Insert(h, eviction_callback);
  }
  virtual Handle* Lookup(const Slice& key, CacheBehavior caching) OVERRIDE {
    const uint32_t hash = HashSlice(key);
    return shards_[Shard(hash)]->Lookup(key, hash, caching == EXPECT_IN_CACHE);
  }
  virtual void Release(Handle* handle) OVERRIDE {
    LRUHandle* h = reinterpret_cast<LRUHandle*>(handle);
    shards_[Shard(h->hash)]->Release(handle);
  }
  virtual void Erase(const Slice& key) OVERRIDE {
    const uint32_t hash = HashSlice(key);
    shards_[Shard(hash)]->Erase(key, hash);
  }
  virtual Slice Value(Handle* handle) OVERRIDE {
    return reinterpret_cast<LRUHandle*>(handle)->value();
  }
  virtual uint8_t* MutableValue(PendingHandle* handle) OVERRIDE {
    return reinterpret_cast<LRUHandle*>(handle)->val_ptr();
  }

  virtual uint64_t NewId() OVERRIDE {
    std::lock_guard<MutexType> l(id_mutex_);
    return ++(last_id_);
  }
  virtual void SetMetrics(const scoped_refptr<MetricEntity>& entity) OVERRIDE {
    metrics_.reset(new CacheMetrics(entity));
    for (NvmLRUCache* cache : shards_) {
      cache->SetMetrics(metrics_.get());
    }
  }
  // Allocate NVM memory. Try until successful or FLAGS_nvm_cache_allocation_retry_count
  // has been exceeded.
  virtual PendingHandle* Allocate(Slice key, size_t val_len, int charge) OVERRIDE {
    size_t key_len = key.size();
    DCHECK_GE(key_len, 0);
    DCHECK_GE(val_len, 0);
    LRUHandle* handle = nullptr;

    for (NvmLRUCache* cache : shards_) {
      // In persistent mode, we allocate the LRUHandle from the heap,
      // but the KV data from pmem using persistent data structures.
      if (IsPersistentMode()) {
        uint8_t* kv_data = reinterpret_cast<uint8_t*>(
            cache->AllocateAndRetry(key_len + val_len,
                                    TOID_TYPE_NUM(struct KeyVal), PMEM_ALLOC,
                                    KvConstructor));
        if (kv_data == nullptr) {
          PLOG_IF(INFO, kv_data == nullptr) << "Could not allocate persistent memory"
            " from current shard. Trying the next shard.";
          continue;
        }
        // Find the full KeyVal structure from the allocation above.
        handle = new LRUHandle; // Handle structure, modulo kv_data comes from DRAM
        handle->repopulated = false;
        handle->kv_data = kv_data; // Pointer to pmem, key and value.
        handle->val_length = val_len;
        handle->key_length = key_len;
        handle->charge = charge + key_len;
        handle->eviction_callback = nullptr;
        memcpy(handle->kv_data, key.data(), key_len);
        handle->hash = HashSlice(key);
      } else {
        // For volatile mode we allocate from cache without a persistent
        // constructor. The buffer is simply a blob of memory with
        // no structure. This memory is used for the LRUHandle buffer.
        uint8_t* buf = static_cast<uint8_t*>(cache->AllocateAndRetry(
            sizeof(LRUHandle) + key_len + val_len, 0, VMEM_ALLOC, nullptr));
        if (buf == nullptr) {
          PLOG_IF(INFO, buf == nullptr) << "Could not allocate persistent memory"
            " from current shard. Trying the next shard.";
            continue;
        }
        handle = reinterpret_cast<LRUHandle*>(buf);
        handle->repopulated = false;
        handle->kv_data = &buf[sizeof(LRUHandle)];
        handle->val_length = val_len;
        handle->key_length = key_len;
        handle->charge = charge;
        handle->eviction_callback = nullptr;
        memcpy(handle->kv_data, key.data(), key.size());
        handle->hash = HashSlice(key);
      }
      return reinterpret_cast<PendingHandle*>(handle);
    }
    // TODO(opt): increment a metric here on allocation failure.
    return nullptr;
  }

  // Free scratch memory. We do not free the pmem memory
  // only the LRUHandle.
  virtual void Free(PendingHandle* ph) OVERRIDE {
    if (IsPersistentMode()) {
      delete ph;
    } else {
      vmem_free(vmp_, ph);
    }
  }
};

// Check the KeyValVer object for a match on major and minor version
bool NvmCheckVersion(struct KeyValVersion* rootp, int major_required, int minor_required) {

  int major;
  int minor;

  // If the nvm_cache_major_version is > 0 then it was set during a
  // runtime test. It's a hidden flag and only used in testing.
  if (FLAGS_nvm_cache_major_version) {
    major = FLAGS_nvm_cache_major_version;
  } else {
    major = rootp->cache_major;
  }
  minor = rootp->cache_minor;

  // The current check is !=. This can be changed to reflect whatever
  // upgrade strategy is decided.
  PLOG_IF(FATAL, major_required != major) << "nvmcache major version"
          << "mismatch (need %u, found %u) " << major;
  PLOG_IF(FATAL, minor_required != minor) << "nvmcache minor version"
          << "mismatch (need %u, found %u) " << minor;
  return (true);
}
} // end anonymous namespace

Cache* NewLRUNvmCache(size_t capacity) {
  VMEM* vmp;
  PMEMobjpool* pop;
  PMEMoid root;

  // Creation of the persistent memory will fail if the capacity is too small,
  // but with an inscrutable error. So, we'll check ourselves.
  if (FLAGS_nvm_cache_persistent) {
    string pmem_path = JoinPathSegments(FLAGS_nvm_cache_path, FLAGS_nvm_cache_pool);
    CHECK_GE(capacity, PMEMOBJ_MIN_POOL)
      << "configured capacity " << capacity << " bytes is less than "
      << "the minimum capacity for an NVM cache: " << PMEMOBJ_MIN_POOL;
    if (access(pmem_path.c_str(), F_OK) != 0) {
      pop = pmemobj_create(pmem_path.c_str(),
                          POBJ_LAYOUT_NAME(kudublockcache),
                          capacity, 0777);
      PLOG_IF(FATAL, pop == NULL) << "Could not initialize NVM cache library in path"
                                    << FLAGS_nvm_cache_path;
      root = pmemobj_root(pop, sizeof(struct KeyValVersion));
      struct KeyValVersion* rootp = static_cast<struct KeyValVersion*>(pmemobj_direct(root));
      rootp->cache_major = NVMCACHE_MAJOR_VERSION;
      rootp->cache_minor = NVMCACHE_MINOR_VERSION;
      pmemobj_persist(pop, rootp, sizeof(struct KeyValVersion));
      vmp = NULL;
    } else {
      pop = pmemobj_open(pmem_path.c_str(),
                         POBJ_LAYOUT_NAME(kudublockcache));
      PLOG_IF(FATAL, pop == NULL) << "Could not initialize NVM cache library in path"
                                    << FLAGS_nvm_cache_path;
      root = pmemobj_root(pop, sizeof(struct KeyValVersion));
      struct KeyValVersion* rootp = static_cast<struct KeyValVersion*>(pmemobj_direct(root));
      bool ver_match = NvmCheckVersion(rootp, NVMCACHE_MAJOR_VERSION,
                                        NVMCACHE_MINOR_VERSION);
      PLOG_IF(FATAL, !ver_match) << "Version mismatch for pool"
                                          << NVMCACHE_MAJOR_VERSION
                                           << NVMCACHE_MINOR_VERSION;

      // In the case where the pool exists, we don't create a new one we
      // simply get the existing one.
      vmp = NULL;
    }
  } else {
    CHECK_GE(capacity, VMEM_MIN_POOL)
      << "configured capacity " << capacity << " bytes is less than "
      << "the minimum capacity for an NVM cache: " << VMEM_MIN_POOL;
    vmp = vmem_create(FLAGS_nvm_cache_path.c_str(), capacity);
    root = OID_NULL;
    pop = NULL;
    // If we cannot create the cache pool we should not retry.
    PLOG_IF(FATAL, vmp == NULL) << "Could not initialize NVM cache library in path "
                                << FLAGS_nvm_cache_path.c_str();
  }
  return new ShardedLRUCache(capacity, pop, root, vmp);
}

} // namespace kudu
