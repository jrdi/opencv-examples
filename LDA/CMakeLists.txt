project( LDA )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(LDA main.cpp)
target_link_libraries( LDA ${OpenCV_LIBS} )
