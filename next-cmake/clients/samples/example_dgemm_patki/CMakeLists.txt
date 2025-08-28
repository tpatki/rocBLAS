# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-dgemm-patki
               ${TEMP_CLIENT_SAMPLES_DIR}/example_dgemm_patki.cpp)

target_link_libraries(rocblas-example-dgemm-patki PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-dgemm-patki PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   ${PROJECT_BINARY_DIR}/staging)
