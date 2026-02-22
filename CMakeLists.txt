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
)

target_include_directories(tinygnn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Tests ───────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE tinygnn_core)
add_test(NAME TensorTests COMMAND test_tensor)
