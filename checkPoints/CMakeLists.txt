project( checkPoints )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(checkPoints main.cpp)
target_link_libraries( checkPoints ${OpenCV_LIBS} )
