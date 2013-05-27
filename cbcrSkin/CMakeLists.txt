project( cbcrSkin )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

add_executable(cbcrSkin main.cpp)
target_link_libraries( cbcrSkin ${OpenCV_LIBS} )
