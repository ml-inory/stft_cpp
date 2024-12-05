mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../ -DCMAKE_BUILD_TYPE=Debug ..
make install