project( example1 )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(example1 main.cpp)
target_link_libraries( example1 ${OpenCV_LIBS} )
