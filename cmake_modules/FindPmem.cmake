# Find Required PMEM libraries (libvmem, libpmem, libpmemobj)
# This module defines
#  PMEM_INCLUDE_DIR, directory containing headers
#  XXX_STATIC_LIBS, path to *.a
#  XXX_SHARED_LIBS, path to *.so shared library
#  PMEM_FOUND, whether PMEM libraries have been found
#  PMEMOBJ_DEPS, dependencies required for using libpmemobj

set(PMEM_SEARCH_LIB_PATH
  ${THIRDPARTY_INSTALL_UNINSTRUMENTED_DIR}/lib
)
set(PMEM_SEARCH_HEADER_PATHS
  ${THIRDPARTY_INSTALL_UNINSTRUMENTED_DIR}/include
)

find_path(VMEM_INCLUDE_DIR libvmem.h PATHS ${PMEM_SEARCH_HEADER_PATHS}
  # make sure we don't accidentally pick up a different version
  NO_DEFAULT_PATH
  )
find_path(PMEM_INCLUDE_DIR libpmem.h PATHS ${PMEM_SEARCH_HEADER_PATHS}
  # make sure we don't accidentally pick up a different version
  NO_DEFAULT_PATH
  )
find_path(PMEMOBJ_INCLUDE_DIR libpmemobj.h PATHS ${PMEM_SEARCH_HEADER_PATHS}
  # make sure we don't accidentally pick up a different version
  NO_DEFAULT_PATH
  )

if (VMEM_INCLUDE_DIR AND PMEM_INCLUDE_DIR AND PMEMOBJ_INCLUDE_DIR)
  find_library(VMEM_LIB_PATH NAMES vmem PATHS ${PMEM_SEARCH_LIB_PATH} NO_DEFAULT_PATH)
  find_library(PMEM_LIB_PATH NAMES pmem PATHS ${PMEM_SEARCH_LIB_PATH} NO_DEFAULT_PATH)
  find_library(PMEMOBJ_LIB_PATH NAMES pmemobj PATHS ${PMEM_SEARCH_LIB_PATH} NO_DEFAULT_PATH)
endif()

if (VMEM_LIB_PATH AND PMEM_LIB_PATH AND PMEMOBJ_LIB_PATH)
  MESSAGE("FOUND ALL PMEM LIB PATHS")
  set(PMEM_FOUND TRUE)
  set(VMEM_LIB_NAME libvmem)
  set(VMEM_STATIC_LIB ${PMEM_SEARCH_LIB_PATH}/${VMEM_LIB_NAME}.a)
  set(VMEM_SHARED_LIB ${PMEM_SEARCH_LIB_PATH}/${VMEM_LIB_NAME}.so)
  set(PMEM_LIB_NAME libpmem)
  set(PMEM_STATIC_LIB ${PMEM_SEARCH_LIB_PATH}/${PMEM_LIB_NAME}.a)
  set(PMEM_SHARED_LIB ${PMEM_SEARCH_LIB_PATH}/${PMEM_LIB_NAME}.so)
  set(PMEMOBJ_LIB_NAME libpmemobj)
  set(PMEMOBJ_STATIC_LIB ${PMEM_SEARCH_LIB_PATH}/${PMEMOBJ_LIB_NAME}.a)
  set(PMEMOBJ_SHARED_LIB ${PMEM_SEARCH_LIB_PATH}/${PMEMOBJ_LIB_NAME}.so)
  set(PMEMOBJ_DEPS pmem)
endif()

if(PMEM_FOUND)
if (NOT PMEM_FIND_QUIETLY)
    message(STATUS "Found the pmem libraries: ${PMEM_SEARCH_LIB_PATH}")
  endif ()
else ()
  if (NOT PMEM_FIND_QUIETLY)
    set(PMEM_ERR_MSG "Could not find the pmemobj library. Looked for headers")
    set(PMEM_ERR_MSG "${PMEM_ERR_MSG} in ${PMEM_SEARCH_HEADER_PATHS}, and for libs")
    set(PMEM_ERR_MSG "${PMEM_ERR_MSG} in ${PMEM_SEARCH_LIB_PATH}")
    if (Pmem_FIND_REQUIRED)
      message(FATAL_ERROR "${PMEM_ERR_MSG}")
    else (Pmem_FIND_REQUIRED)
      message(STATUS "${PMEM_ERR_MSG}")
    endif (Pmem_FIND_REQUIRED)
  endif ()
endif ()

mark_as_advanced(
  PMEM_INCLUDE_DIR
  PMEMOBJ_DEPS
  VMEM_STATIC_LIB
  VMEM_SHARED_LIB
  PMEM_STATIC_LIB
  PMEM_SHARED_LIB
  PMEMOBJ_STATIC_LIB
  PMEMOBJ_SHARED_LIB
)

