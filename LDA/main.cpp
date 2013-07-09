#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

cv::Mat lda(cv::Mat & X, std::vector<int> y) {
  int dimensions = X.cols;
  cv::Mat Sw = cv::Mat::zeros(dimensions, dimensions, CV_32F);
  cv::Mat Sb = cv::Mat::zeros(dimensions, dimensions, CV_32F);

  cv::Mat Covar, Mu;
  cv::calcCovarMatrix(X, Covar, Mu, CV_COVAR_NORMAL+CV_COVAR_ROWS, CV_32F);

  std::cout << "Mu: " << Mu << std::endl;

  cv::Mat X0, X1;
  for (int i = 0; i < y.size(); ++i)
  {
    if(y[i] == 0) {
      X0.push_back(X.row(i));
    } else if(y[i] == 1) {
      X1.push_back(X.row(i));
    }
  }

  int n0 = X0.rows;
  int n1 = X1.rows;

  cv::Mat Covar0, Covar1, Mu0, Mu1;
  cv::calcCovarMatrix(X0, Covar0, Mu0, CV_COVAR_NORMAL+CV_COVAR_ROWS, CV_32F);
  cv::calcCovarMatrix(X1, Covar1, Mu1, CV_COVAR_NORMAL+CV_COVAR_ROWS, CV_32F);

  std::cout << "Bg Mu: " << Mu0 << std::endl;
  std::cout << "Skin Mu: " << Mu1 << std::endl;

  cv::Mat XM0(X0.size(), X0.type()), XM1(X1.size(), X1.type());
  for(int i = 0; i < X0.rows; i++) {
    for(int j = 0; j < dimensions; j++) {
      XM0.at<float>(i, j) = X0.at<float>(i, j) - Mu0.at<float>(j);
    }
  }

  for(int i = 0; i < X1.rows; i++) {
    for(int j = 0; j < dimensions; j++) {
      XM1.at<float>(i, j) = X1.at<float>(i, j) - Mu1.at<float>(j);
    }
  }

  Sw = (XM0.t() * XM0) + (XM1.t() * XM1);
  std::cout << "Sw: " << Sw << std::endl;
  
  cv::Mat M0M(dimensions, 1, CV_32F);
  for(int j = 0; j < dimensions; j++) {
    M0M.at<float>(j) = Mu0.at<float>(j) - Mu.at<float>(j);
  }

  cv::Mat M1M(dimensions, 1, CV_32F);
  for(int j = 0; j < dimensions; j++) {
    M1M.at<float>(j) = Mu1.at<float>(j) - Mu.at<float>(j);
  }

  Sb = n0 * M0M * M0M.t() + n1 * M1M * M1M.t();
  std::cout << "Sb: " << Sb << std::endl;

  cv::Mat S = Sw.inv(cv::DECOMP_SVD) * Sb;
  std::cout << "S: " << S << std::endl;

  cv::Mat eigenvalues(1, dimensions, Sb.type()), eigenvectors(dimensions, dimensions, Sb.type());
  cv::eigen(S, eigenvalues, eigenvectors);

  std::cout << eigenvalues << std::endl;
  std::cout << eigenvectors << std::endl;

  return eigenvectors;
}

int main( int argc, char** argv )
{
  if( argc != 4 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <bgImage> <skinImage>" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat image = cv::imread( argv[1], 1 );
  if(!image.data) {
    return EXIT_FAILURE;
  }
  cv::Mat bgImage = cv::imread( argv[2], 1 );
  if(!bgImage.data) {
    return EXIT_FAILURE;
  }
  cv::Mat skinImage = cv::imread( argv[3], 1 );
  if(!skinImage.data) {
    return EXIT_FAILURE;
  }

  cv::Mat samples(bgImage.cols*bgImage.rows+skinImage.cols*skinImage.rows, 2, CV_32F);
  std::vector<int> labels(bgImage.cols*bgImage.rows+skinImage.cols*skinImage.rows);

  bgImage.convertTo(bgImage, CV_32F);
  skinImage.convertTo(skinImage, CV_32F);

  cv::cvtColor( bgImage, bgImage, CV_BGR2YCrCb );
  cv::cvtColor( skinImage, skinImage, CV_BGR2YCrCb );

  int count = 0;

  for( int y = 0; y < bgImage.rows ; y++ ) { 
    for( int x = 0; x < bgImage.cols; x++ ) {
      cv::Vec3f color = bgImage.at<cv::Vec3f>(y, x);

      samples.at<float>(count, 0) = color[1];
      samples.at<float>(count, 1) = color[2];

      labels[count] = 0;

      count++;
    }
  }

  for( int y = 0; y < skinImage.rows ; y++ ) { 
    for( int x = 0; x < skinImage.cols; x++ ) {
      cv::Vec3f color = skinImage.at<cv::Vec3f>(y, x);

      samples.at<float>(count, 0) = color[1];
      samples.at<float>(count, 1) = color[2];

      labels[count] = 1;

      count++;
    }
  }

  std::ofstream ofs;
  ofs.open( "myFile.txt" );

  for( int i = 0; i < samples.rows; i++) {
    ofs << samples.at<float>(i, 0) << " " << samples.at<float>(i, 1) << std::endl;
  }

  ofs.close();

  cv::Mat convMat = lda(samples, labels);

  image.convertTo(image, CV_32F);
  cv::cvtColor( image, image, CV_BGR2YCrCb );
  cv::imwrite("cbcr.jpg", image);

  count = 0;
  for( int y = 0; y < image.rows ; y++ ) { 
    for( int x = 0; x < image.cols; x++ ) {
      cv::Vec3f color = image.at<cv::Vec3f>(y, x);
      cv::Mat sample(2, 1, CV_32F), resample(2, 1, CV_32F);

      sample.at<float>(0) = color[1];
      sample.at<float>(1) = color[2];

      resample = convMat * sample;
      image.at<cv::Vec3f>(y, x) = cv::Vec3f(color[0], resample.at<float>(0), resample.at<float>(1));

      count++;
    }
  }

  cv::imwrite("cbcr2.jpg", image);

  return EXIT_SUCCESS;
}

