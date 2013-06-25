#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

cv::Mat image;
int thresh = 15;
int max_thresh = 50;

int intSigmaBig = 70;
int intMaxSigmaBig = 120;

int intSigmaSmall = 60;
int intMaxSigmaSmall = 120;

char* source_window = "Source image";

void filterDoG( int, void* );

int main( int argc, char** argv )
{
  if( argc != 2 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage>" << std::endl;
    return EXIT_FAILURE;
  }

  image = cv::imread( argv[1], 1 );
  if(!image.data) {
    return EXIT_FAILURE;
  }

  cv::namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  cv::createTrackbar( "sigmaBig: ", source_window, &intSigmaBig, intMaxSigmaBig, filterDoG );
  cv::createTrackbar( "sigmaSmall: ", source_window, &intSigmaSmall, intMaxSigmaSmall, filterDoG );
  cv::createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, filterDoG );
  cv::imshow( source_window, image );

  filterDoG(0, 0);

  cv::waitKey(0);
  return EXIT_SUCCESS;
}

void filterDoG( int, void* )
{
  cv::Mat filterResponse;
  float sigmaBig = intSigmaBig / 10.0f;
  float sigmaSmall = intSigmaSmall / 100.0f;

  // sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
  int ksize = ceilf((sigmaBig-0.8f)/0.3f)*2 + 3;

  cv::Mat gauBig = cv::getGaussianKernel(ksize, sigmaBig, CV_32F);
  cv::Mat gauSmall = cv::getGaussianKernel(ksize, sigmaSmall, CV_32F);

  cv::Mat DoG = gauSmall - gauBig;
  cv::sepFilter2D(image, filterResponse, CV_32F, DoG.t(), DoG);

  filterResponse = cv::abs(filterResponse);

  std::cout << thresh << " : " << sigmaBig << " : " << sigmaSmall << std::endl;

  cv::Mat dst = image.clone();

  for( int j = 0; j < filterResponse.rows ; j++ ) { 
    for( int i = 0; i < filterResponse.cols; i++ ) {
      cv::Vec3f absPixel  = filterResponse.at<cv::Vec3f>(j, i);

      if( (absPixel[0]+absPixel[1]+absPixel[2])/3 >= thresh ) {
        circle( dst, cv::Point( i, j ), 1, cv::Scalar(0, 0, 255), 2, 8, 0 );
      }
    }
  }

  cv::resize(dst, dst, cv::Size(round(700 * dst.cols/dst.rows), 700));

  cv::namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( source_window, dst );
}

