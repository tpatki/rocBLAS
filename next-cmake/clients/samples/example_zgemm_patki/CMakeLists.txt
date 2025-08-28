# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-zgemm-patki
               ${TEMP_CLIENT_SAMPLES_DIR}/example_zgemm_patki.cpp)

target_link_libraries(rocblas-example-zgemm-patki PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-zgemm-patki PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   ${PROJECT_BINARY_DIR}/staging)
