# ============================================================================
#  TinyGNN vcpkg port  —  portfile.cmake
#
#  After pushing the v0.1.3 release tag to GitHub, compute SHA512:
#
#    Invoke-WebRequest -Uri https://github.com/AnubhavChoudheries/TinyGNN/archive/refs/tags/v0.1.3.tar.gz -OutFile v013.tar.gz
#    (Get-FileHash v013.tar.gz -Algorithm SHA512).Hash.ToLower()
#
#  Replace the SHA512 placeholder below with that output.
# ============================================================================

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO AnubhavChoudhery/TinyGNN
    REF "v${VERSION}"
    SHA512 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DTINYGNN_BUILD_TESTS=OFF
        -DTINYGNN_BUILD_BENCHMARKS=OFF
)

vcpkg_cmake_install()

# Move the CMake config files from datadir → share/<port> (vcpkg convention)
vcpkg_cmake_config_fixup(PACKAGE_NAME tinygnn CONFIG_PATH share/tinygnn)

# Remove the debug include tree (headers are not debug-specific)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# Install license (vcpkg requires a file named 'copyright')
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
