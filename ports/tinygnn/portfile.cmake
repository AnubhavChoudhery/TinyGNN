# ============================================================================
#  TinyGNN vcpkg port  —  portfile.cmake
#
#  Step 1 (one-time):  Create a GitHub release tagged v0.1.2 and upload
#                      the source tarball.  Then compute its SHA512:
#
#    vcpkg hash <downloaded-tarball>
#
#  Replace the SHA512 below with the output of that command, then submit
#  a PR to microsoft/vcpkg.
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
