project( maskedImage )

cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set( maskedImageExe ${CMAKE_BINARY_DIR}/maskedImage )
add_subdirectory(scripts)


add_executable(maskedImage main.cpp)
target_link_libraries( maskedImage ${OpenCV_LIBS} )
