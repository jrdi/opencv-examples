#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

cv::Mat image;
int thresh = 0;
int max_thresh = 100;
bool useDet = true;
int gauKsize = 11;

char* source_window = "Source image";

void filterHessian( int, void* );

int main( int argc, char** argv )
{
  if( argc != 4 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <method> <OutputFolder>" << std::endl;
    return EXIT_FAILURE;
  }

  image = cv::imread( argv[1], 1 );
  if(!image.data) {
    return EXIT_FAILURE;
  }

  useDet = (atoi(argv[2]) == 1);

  cv::namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  cv::createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, filterHessian );
  cv::imshow( source_window, image );

  filterHessian(0, 0);

  cv::waitKey(0);
  return EXIT_SUCCESS;
}

void filterHessian( int, void* )
{
  cv::Mat dXX, dYY, dXY;
  std::vector<cv::KeyPoint> keypoints;

  cv::Sobel(image, dXX, CV_32F, 2, 0);
  cv::Sobel(image, dYY, CV_32F, 0, 2);
  cv::Sobel(image, dXY, CV_32F, 1, 1);
  
  cv::Mat gau = cv::getGaussianKernel(gauKsize, -1, CV_32F);
  
  cv::sepFilter2D(dXX, dXX, CV_32F, gau.t(), gau);
  cv::sepFilter2D(dYY, dYY, CV_32F, gau.t(), gau);
  cv::sepFilter2D(dXY, dXY, CV_32F, gau.t(), gau);
  
  cv::Mat dst = image.clone();
  
  std::cout << thresh << std::endl;

  if(useDet) {
    cv::Mat detH = dXX.mul(dYY) - dXY.mul(dXY);

    for( int j = 0; j < detH.rows ; j++ ) { 
      for( int i = 0; i < detH.cols; i++ ) {
        cv::Vec3f pixelHessianResponse = detH.at<cv::Vec3f>(j, i);

        if( (pixelHessianResponse[0]+pixelHessianResponse[1]+pixelHessianResponse[2])/3 >= thresh ) {
          circle( dst, cv::Point( i, j ), 1, cv::Scalar(0, 0, 255), 2, 8, 0 );
        }
      }
    }
  }
  else
  {
    cv::Mat matrix(2, 2, CV_32F);
    std::vector<float> eigenvalues(2);
    
    for( int j = 0; j < image.rows ; j++ ) { 
      for( int i = 0; i < image.cols; i++ ) {
        matrix.at<float>(0, 0) = dXX.at<float>(j, i);
        matrix.at<float>(1, 1) = dYY.at<float>(j, i);
        matrix.at<float>(0, 1) = matrix.at<float>(1, 0) = dXY.at<float>(j, i);

        eigen(matrix, eigenvalues);

        float ratio = ((abs(eigenvalues[0]) > abs(eigenvalues[1])) ?
                       eigenvalues[0] / eigenvalues[1] : eigenvalues[1] / eigenvalues[0]);
      
        if(ratio > 0.0f && ratio < thresh) {
          circle( dst, cv::Point( i, j ), 1, cv::Scalar(0, 0, 255), 2, 8, 0 );
        }
      }
    }
  }

  cv::namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( source_window, dst );
}

