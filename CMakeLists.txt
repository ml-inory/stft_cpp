project(stft)
cmake_minimum_required(VERSION 3.10)

include_directories(src)
add_executable(stft main.cpp)
install(TARGETS stft
    RUNTIME DESTINATION ./)