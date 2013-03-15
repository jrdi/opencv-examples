#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

cv::Mat image;
int thresh = 65;
int max_thresh = 150;

int gauKsize = 11;

char* source_window = "Source image";

void filterGradient( int, void* );

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
  cv::createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, filterGradient );
  cv::imshow( source_window, image );

  filterGradient(0, 0);

  cv::waitKey(0);
  return EXIT_SUCCESS;
}

void filterGradient( int, void* )
{
  cv::Mat dX, dY;

  cv::Sobel(image, dX, CV_32F, 1, 0);
  cv::Sobel(image, dY, CV_32F, 0, 1);

  // Gaussian kernel with ksize = gauKsize and sigma = -1 (autocomputed internally)
  cv::Mat gau = cv::getGaussianKernel(gauKsize, -1, CV_32F);

  cv::sepFilter2D(dX, dX, CV_32F, gau.t(), gau);
  cv::sepFilter2D(dY, dY, CV_32F, gau.t(), gau);

  cv::Mat dst = image.clone();
  cv::Mat grad(image.size(), image.type());

  std::cout << thresh << std::endl;

  for( int j = 0; j < image.rows ; j++ ) { 
    for( int i = 0; i < image.cols; i++ ) {
      cv::Vec3f pixelX = dX.at<cv::Vec3f>(j, i);
      cv::Vec3f pixelY = dY.at<cv::Vec3f>(j, i);



      grad.at<cv::Vec3f>(j, i)[0] = cv::sqrt(cv::pow(pixelX[0], 2) + cv::pow(pixelY[0], 2));
      grad.at<cv::Vec3f>(j, i)[1] = cv::sqrt(cv::pow(pixelX[1], 2) + cv::pow(pixelY[1], 2));
      grad.at<cv::Vec3f>(j, i)[2] = cv::sqrt(cv::pow(pixelX[2], 2) + cv::pow(pixelY[2], 2));

      if( grad.at<cv::Vec3f>(j, i)[0] > thresh && grad.at<cv::Vec3f>(j, i)[1] > thresh && grad.at<cv::Vec3f>(j, i)[2] > thresh ) {
        circle( dst, cv::Point( i, j ), 1, cv::Scalar(0, 0, 255), 2, 8, 0 );
      }
    }
  }

  cv::namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( source_window, dst );
}

