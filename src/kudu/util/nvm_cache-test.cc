// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <libpmemobj.h>
#include <memory>
#include <string>
#include <vector>

#include "kudu/util/cache.h"
#include "kudu/util/coding.h"
#include "kudu/util/env.h"
#include "kudu/util/mem_tracker.h"
#include "kudu/util/metrics.h"
#include "kudu/util/test_util.h"

#if defined(__linux__)
#define NVM_CACHE_ENABLED
#endif

#if defined(NVM_CACHE_ENABLED)
DECLARE_string(nvm_cache_path);
DECLARE_bool(nvm_cache_persistent);
DECLARE_string(nvm_cache_directory);
DECLARE_int32(nvm_cache_major_version);

namespace kudu {

using std::string;

// Conversions between numeric keys/values and the types expected by Cache.
static std::string EncodeInt(int k) {
  faststring result;
  PutFixed32(&result, k);
  return result.ToString();
}

static int DecodeInt(const Slice& k) {
  assert(k.size() == 4);
  return DecodeFixed32(k.data());
}

class NvmCacheTest : public KuduTest,
                     public ::testing::WithParamInterface<CacheType> {
 public:
  gscoped_ptr<Cache> cache_;
  static const int kCacheSize = 14*1024*1024;
  static const int kNumElems = 10;
  const int kSizePerElem = kCacheSize / kNumElems;
  string PmemPath;

  virtual void SetUp() OVERRIDE {
    // We are only testing the persistent mode pmem cache.
    if (google::GetCommandLineFlagInfoOrDie("nvm_cache_path").is_default) {
      FLAGS_nvm_cache_path = GetTestPath("nvm-cache");
      PmemPath = FLAGS_nvm_cache_path;

      // This test relies on the fact that the original persistent memory
      // pool is available after the first test. Don't create the directory
      // if it's always there.
      if (access(PmemPath.c_str(), F_OK != 0)) {
        Env::Default()->CreateDir(FLAGS_nvm_cache_path);
      }
    }

    switch (GetParam()) {
      case NVM_CACHE_PERSISTENT:
        break;
      default:
        LOG(FATAL) << "Unknown block cache type: '" << GetParam();
    }
    cache_.reset(NewLRUCache(GetParam(), kCacheSize, "cache_test"));
  }

  int Lookup(int key) {
    Cache::Handle* handle = cache_->Lookup(EncodeInt(key), Cache::EXPECT_IN_CACHE);
    const int r =
      (handle == nullptr) ? -1 : DecodeInt(cache_->Value(handle));
    if (handle != nullptr) {
      cache_->Release(handle);
    }
    return r;
  }

  // Insert entry into cache. May fail based on the error simulation
  // flags settings.
  void Insert(int key, int value, int charge = 1) {
    string key_str = EncodeInt(key);
    string val_str = EncodeInt(value);
    Cache::PendingHandle* phandle =
      cache_->Allocate(key_str, val_str.size(), charge);
    memcpy(cache_->MutableValue(phandle), val_str.data(), val_str.size());
    cache_->Release(cache_->Insert(phandle, nullptr));
  }

};

#if defined(NVM_CACHE_ENABLED)
INSTANTIATE_TEST_CASE_P(NvmCacheTypes, NvmCacheTest,
                        ::testing::Values(NVM_CACHE_PERSISTENT));
#endif

// Between these tests the cache is closed and reopened, thus verifying
// the persistence of the data in the cache.
TEST_P(NvmCacheTest, TestCachePersistence) {
  for (int i = 0; i < kNumElems + 100; i++) {
    Insert(1000 + i, 2000 + i, kSizePerElem);
  }

  // Close the pool. This forces a reload of the cache and we
  // can check to see if there is any corruption. XXX: The better
  // solution would be to simulate a failure during the process of
  // creation, persisting and insertion into the cache.
  cache_.reset();
  cache_.reset(NewLRUCache(GetParam(), kCacheSize, "cache_test"));

  // Look up to verify the cache contents
  for (int i = 0; i < kNumElems + 100; i ++) {
    ASSERT_EQ(2000 + i, Lookup(1000 + i));
  }
}

TEST_P(NvmCacheTest, TestCacheVersion) {
  Insert(100, 1, kSizePerElem);
  cache_.reset();

  // Cache was originally created with value defined as NVMCACHE_MAJOR_VERSION. Set
  // the hidden runtime flag to something other than current major and test
  // for failure to open.
  cache_.reset(NewLRUCache(GetParam(), kCacheSize, "cache_test"));
  cache_.reset();
  FLAGS_nvm_cache_major_version = FLAGS_nvm_cache_major_version + 1;
  ASSERT_EXIT(cache_.reset(NewLRUCache(GetParam(), kCacheSize, "cache_test")),
             ::testing::KilledBySignal(6), "nvmcache major version");
}


} // namespace kudu
#endif
