project(stft)
cmake_minimum_required(VERSION 3.10)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -fsanitize=address")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2 -fsanitize=address")
endif()

include_directories(src)
add_executable(stft main.cpp)
install(TARGETS stft
    RUNTIME DESTINATION ./)