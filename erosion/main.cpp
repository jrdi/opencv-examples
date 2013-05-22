#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char** argv ) {

  if( argc != 2 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage>" << std::endl;
    return EXIT_FAILURE;
  }
  cv::Mat image;
  image = cv::imread( argv[1], 1 );
  if(!image.data) {
    return EXIT_FAILURE;
  }

  cv::Mat grayImage(image.size(), CV_8U);
  cv::cvtColor( image, grayImage, CV_BGR2GRAY );
  cv::Mat binaryImage(grayImage.size(), grayImage.type());
  cv::threshold(grayImage, binaryImage, 1, 255, cv::THRESH_BINARY);

  int erosion_size = 4;
  cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
                                          cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ));

  cv::Mat roi(binaryImage.size(), binaryImage.type());
  cv::erode( binaryImage, roi, element );

  cv::cvtColor( roi, roi, CV_GRAY2BGR );

  cv::Mat dst(image.size(), image.type());
  cv::bitwise_and(image, roi, dst);

  cv::imwrite("image.jpg", dst);

  return EXIT_SUCCESS;
}