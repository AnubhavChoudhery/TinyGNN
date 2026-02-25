cmake_minimum_required(VERSION 3.16)
project(TinyGNN VERSION 0.1.0 LANGUAGES CXX)

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

# ── Core library ────────────────────────────────────────────────────────────
add_library(tinygnn_core
    src/tensor.cpp
    src/graph_loader.cpp
    src/ops.cpp
    src/layers.cpp
    src/model.cpp
)

target_include_directories(tinygnn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Tests ───────────────────────────────────────────────────────────────────
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
add_test(NAME E2ETests COMMAND test_e2e)
