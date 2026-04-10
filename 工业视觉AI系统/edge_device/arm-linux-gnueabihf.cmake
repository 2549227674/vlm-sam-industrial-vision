# CMake Toolchain File for i.MX6ULL (ARM Cortex-A7)
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=../arm-linux-gnueabihf.cmake ..
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Cross compiler
set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)

# Sysroot (if using a custom sysroot, uncomment and modify)
# set(CMAKE_SYSROOT /path/to/your/sysroot)

# Search paths for libraries and headers
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Compiler flags for ARM Cortex-A7 (i.MX6ULL)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")

