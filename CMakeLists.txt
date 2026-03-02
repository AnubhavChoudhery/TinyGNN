cmake_minimum_required(VERSION 3.16)
project(TinyGNN VERSION 0.1.2 LANGUAGES CXX)

# ── C++17 standard ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ──────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # Static link on MinGW to avoid ld dynamic linking issues
    if(MINGW)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    endif()
endif()

# ── Build options ──────────────────────────────────────────────────────────
option(TINYGNN_BUILD_TESTS      "Build TinyGNN test executables"      ON)
option(TINYGNN_BUILD_BENCHMARKS "Build TinyGNN benchmark executables" ON)

# ── Core library ────────────────────────────────────────────────────────────
add_library(tinygnn_core
    src/tensor.cpp
    src/graph_loader.cpp
    src/ops.cpp
    src/layers.cpp
    src/model.cpp
)

# Namespaced alias so consumers use tinygnn::tinygnn_core
add_library(tinygnn::tinygnn_core ALIAS tinygnn_core)

target_include_directories(tinygnn_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── OpenMP (Phase 8) ───────────────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tinygnn_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── AVX2 + FMA intrinsics (Phase 8) ────────────────────────────────────────
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma"  COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    target_compile_options(tinygnn_core PRIVATE -mavx2 -mfma)
endif()

# ── Tests ───────────────────────────────────────────────────────────────────
if(TINYGNN_BUILD_TESTS)
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE tinygnn_core)
add_test(NAME TensorTests COMMAND test_tensor)

add_executable(test_graph_loader tests/test_graph_loader.cpp)
target_link_libraries(test_graph_loader PRIVATE tinygnn_core)
add_test(NAME GraphLoaderTests COMMAND test_graph_loader)

add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE tinygnn_core)
add_test(NAME MatmulTests COMMAND test_matmul)

add_executable(test_spmm tests/test_spmm.cpp)
target_link_libraries(test_spmm PRIVATE tinygnn_core)
add_test(NAME SpmmTests COMMAND test_spmm)

add_executable(test_activations tests/test_activations.cpp)
target_link_libraries(test_activations PRIVATE tinygnn_core)
add_test(NAME ActivationsTests COMMAND test_activations)

add_executable(test_gcn tests/test_gcn.cpp)
target_link_libraries(test_gcn PRIVATE tinygnn_core)
add_test(NAME GCNTests COMMAND test_gcn)

add_executable(test_graphsage tests/test_graphsage.cpp)
target_link_libraries(test_graphsage PRIVATE tinygnn_core)
add_test(NAME GraphSAGETests COMMAND test_graphsage)

add_executable(test_gat tests/test_gat.cpp)
target_link_libraries(test_gat PRIVATE tinygnn_core)
add_test(NAME GATTests COMMAND test_gat)

add_executable(test_e2e tests/test_e2e.cpp)
target_link_libraries(test_e2e PRIVATE tinygnn_core)
add_test(NAME E2ETests COMMAND test_e2e
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif() # TINYGNN_BUILD_TESTS

# ── Benchmarks ─────────────────────────────────────────────────────────────
if(TINYGNN_BUILD_BENCHMARKS)
add_executable(bench_parallel benchmarks/bench_parallel.cpp)
target_link_libraries(bench_parallel PRIVATE tinygnn_core)

add_executable(bench_fusion benchmarks/bench_fusion.cpp)
target_link_libraries(bench_fusion PRIVATE tinygnn_core)

add_executable(bench_datasets benchmarks/bench_datasets.cpp)
target_link_libraries(bench_datasets PRIVATE tinygnn_core)
endif() # TINYGNN_BUILD_BENCHMARKS

# ── Installation & CMake package export ────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS tinygnn_core
    EXPORT tinygnn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export the targets file
install(EXPORT tinygnn-targets
    FILE tinygnn-targets.cmake
    NAMESPACE tinygnn::
    DESTINATION ${CMAKE_INSTALL_DATADIR}/tinygnn
)

# Generate config + version files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tinygnn-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tinygnn-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/tinygnn
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tinygnn-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tinygnn-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tinygnn-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/tinygnn
)

# Install the license for downstream consumers
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/tinygnn
    RENAME copyright
)
