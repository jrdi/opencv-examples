#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

cv::Mat image;
int thresh1 = 106;
int max_thresh1 = 500;

int thresh2 = 140;
int max_thresh2 = 500;

int aperture = 3;
int max_aperture = 60;

char* source_window = "Source image";

void filterCanny( int, void* );

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
  cv::createTrackbar( "Threshold 1: ", source_window, &thresh1, max_thresh1, filterCanny );
  cv::createTrackbar( "Threshold 2: ", source_window, &thresh2, max_thresh2, filterCanny );
  cv::createTrackbar( "Aperture: ", source_window, &aperture, max_aperture, filterCanny );
  cv::imshow( source_window, image );

  filterCanny(0, 0);

  cv::waitKey(0);
  return EXIT_SUCCESS;
}

void filterCanny( int, void* )
{
  cv::Mat canny; 
  cv::Mat dst;

  cvtColor(image, dst, CV_RGB2YCrCb);
  std::vector<cv::Mat> splitted;
  cv::split(dst, splitted);
  std::vector<cv::Mat> CbCr(2);
  CbCr[0] = splitted[1];
  CbCr[1] = splitted[2];

  cv::merge(CbCr, dst);

  cv::Canny(image, canny, thresh1, thresh2, aperture, false );

  std::cout << thresh1 << " : " << thresh2 << " : " << aperture << std::endl;

  for( int j = 0; j < canny.rows ; j++ ) { 
    for( int i = 0; i < canny.cols; i++ ) {
      uchar pixel = canny.at<uchar>(j, i);

      if( pixel > uchar(10) ) {
        circle( dst, cv::Point( i, j ), 1, cv::Scalar(0, 0, 255), 2, 8, 0 );
      }
    }
  }

  cv::resize(canny, canny, cv::Size(round(700 * canny.cols/canny.rows), 700));

  cv::namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( source_window, canny );
}

