project( hessian )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(hessian main.cpp)
target_link_libraries( hessian ${OpenCV_LIBS} )
