project(stft)

include_directories(src)
add_executable(stft main.cpp)
install(TARGETS stft
    RUNTIME DESTINATION ./)