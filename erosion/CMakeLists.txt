project( erosion )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(erosion main.cpp)
target_link_libraries( erosion ${OpenCV_LIBS} )
