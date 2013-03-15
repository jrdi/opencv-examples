project( gradient )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(gradient main.cpp)
target_link_libraries( gradient ${OpenCV_LIBS} )
