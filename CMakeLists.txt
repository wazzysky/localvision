cmake_minimum_required(VERSION 3.16)
project(SynchronizedStereoVision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的库
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 添加头文件目录
include_directories(
    ${LIBCAMERA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 定义可执行文件及其源文件
add_executable(stereo_vision
    main.cpp
    detector.cpp
    detector.h
    transform.cpp
    transform.h
    spi_device.cpp
    spi_device.h
)

# 链接库
target_link_libraries(stereo_vision
    ${OpenCV_LIBS}
    ${LIBCAMERA_LIBRARIES}
    pthread
)

# 设置C++编译器标志
set_target_properties(stereo_vision PROPERTIES
    CXX_EXTENSIONS OFF
)