project( LDA )

cmake_minimum_required( VERSION 2.8 )

set( CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${CMAKE_MODULE_PATH}
)

find_package( OpenCV REQUIRED )

find_package( Eigen3 REQUIRED )
INCLUDE_DIRECTORIES( BEFORE ${EIGEN3_INCLUDE_DIR} )

add_executable(LDA main.cpp)
target_link_libraries( LDA 
  ${OpenCV_LIBS} 
  ${Eigen3_LIBS} 
)
