#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char** argv ) {

  if( argc != 3 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <TrainImage>" << std::endl;
    return EXIT_FAILURE;
  }
  cv::Mat image;
  image = cv::imread( argv[1] );
  if(!image.data) {
    return EXIT_FAILURE;
  }
  cv::Mat trainImage;
  trainImage = cv::imread( argv[2] );
  if(!trainImage.data) {
    return EXIT_FAILURE;
  }

  cv::Mat CbCrImage;
  cv::cvtColor( image, CbCrImage, CV_BGR2YCrCb );
  cv::cvtColor( trainImage, trainImage, CV_BGR2YCrCb );

  CbCrImage.convertTo(CbCrImage, CV_64F);
  trainImage.convertTo(trainImage, CV_64F);

  cv::Mat trainSamples = cv::Mat(trainImage.rows*trainImage.cols, 2, CV_64F);

  for( int j = 0; j < trainImage.rows ; j++ ) { 
    for( int i = 0; i < trainImage.cols; i++ ) {
      cv::Vec3d color = trainImage.at<cv::Vec3d>(j, i);

      trainSamples.at<double>(j*i, 0) = color[1];
      trainSamples.at<double>(j*i, 1) = color[2];
    }
  }

  int nClusters = 2;
  cv::Mat means(2, 2, CV_64FC1);  
  means.at<double>(0, 0) = 0;
  means.at<double>(0, 1) = 0;
  means.at<double>(1, 0) = 153.;
  means.at<double>(1, 1) = 100.;

  cv::Mat sample(1, 2, CV_64F);
  cv::Mat mask1(CbCrImage.size(), CV_64FC1);
  cv::Mat mask2(CbCrImage.size(), CV_32FC1);

  cv::EM em_skin = cv::EM(nClusters);
  if(em_skin.trainE(trainSamples, means)) {
    for( int j = 0; j < CbCrImage.rows ; j++ ) { 
      for( int i = 0; i < CbCrImage.cols; i++ ) {
        cv::Vec3d color = CbCrImage.at<cv::Vec3d>(j, i);

        sample.at<double>(0) = color[1];
        sample.at<double>(1) = color[2];

        cv::Vec2d prob = em_skin.predict(sample);
        mask1.at<double>(j, i, 0) = prob[0];
        mask2.at<double>(j, i, 0) = prob[1];
      }
    }
  } else {
    std::cerr << "Error!" << std::endl;
  }

  std::cout << "Initial params: " << std::endl;
  std::cout << "Clusters: " << nClusters << std::endl;
  std::cout << "Means: " << means << std::endl;
  std::cout << "Trained params: " << std::endl;
  std::cout << "Weights: " << em_skin.get<cv::Mat>("weights") << std::endl;
  std::cout << "Means: " << em_skin.get<cv::Mat>("means") << std::endl;
  std::cout << "Covs: " << std::endl;
  std::vector<cv::Mat> covs = em_skin.get<std::vector<cv::Mat> >("covs");
  for (int i = 0; i < nClusters; ++i) {
    std::cout << covs[i] << std::endl;
  }
  std::cout << "Iterations: " << em_skin.get<int>("maxIters") << std::endl;

  double min, max;
  cv::minMaxLoc(mask1, &min, &max);
  std::cout << min << " " << max << std::endl;

  mask1.convertTo(mask1, CV_32FC1);
  cv::threshold(mask1, mask2, -40., 255., cv::THRESH_BINARY);

  cv::normalize(mask1, mask1, 0., 255., cv::NORM_MINMAX);
  cv::imwrite("mask1.jpg", mask1);

  int erosion_size = 3;
  cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
                                          cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ));

  cv::dilate( mask2, mask2, element );
  cv::erode( mask2, mask2, element );

  cv::imwrite("mask2.jpg", mask2);

  return EXIT_SUCCESS;
}