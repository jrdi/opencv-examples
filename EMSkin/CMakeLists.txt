project( EMSkin )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(EMSkin main.cpp)
target_link_libraries( EMSkin ${OpenCV_LIBS} )
