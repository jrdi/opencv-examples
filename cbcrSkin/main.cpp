#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char** argv ) {

  if( argc != 3 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <outputMask>" << std::endl;
    return EXIT_FAILURE;
  }
  cv::Mat image;
  image = cv::imread( argv[1] );
  if(!image.data) {
    return EXIT_FAILURE;
  }

  cv::Mat CbCrImage;
  cv::cvtColor( image, CbCrImage, CV_BGR2YCrCb );

  cv::Mat CbCrMask(image.size(), CV_8U);

  for( int j = 0; j < CbCrImage.rows ; j++ ) { 
    for( int i = 0; i < CbCrImage.cols; i++ ) {
      cv::Vec3b pixelCbCr = CbCrImage.at<cv::Vec3b>(j, i);

      if( pixelCbCr[1] >= 133 && pixelCbCr[2] >= 80 && pixelCbCr[1] <= 173 && pixelCbCr[2] <= 120) {
        CbCrMask.at<uchar>(j, i) = 255;
      } else {
        CbCrMask.at<uchar>(j, i) = 0;
      }
    }
  }

  int erosion_size = 4;
  cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
                                          cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ));

  cv::dilate( CbCrMask, CbCrMask, element );
  cv::erode( CbCrMask, CbCrMask, element );

  cv::imwrite(argv[2], CbCrMask);

  return EXIT_SUCCESS;
}