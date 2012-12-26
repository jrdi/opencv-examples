project( Proto )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(example1 main.cpp)
target_link_libraries( example1 ${OpenCV_LIBS} )
