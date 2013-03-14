project( canny )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(canny main.cpp)
target_link_libraries( canny ${OpenCV_LIBS} )
