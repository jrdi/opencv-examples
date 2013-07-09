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
  cv::Mat Sw = cv::Mat::zeros(dimensions, dimensions, CV_64F);
  cv::Mat Sb = cv::Mat::zeros(dimensions, dimensions, CV_64F);

  cv::Mat Covar, Mu;
  cv::calcCovarMatrix(X, Covar, Mu, CV_COVAR_NORMAL+CV_COVAR_ROWS, CV_64F);

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
  cv::calcCovarMatrix(X0, Covar0, Mu0, CV_COVAR_NORMAL+CV_COVAR_ROWS, CV_64F);
  cv::calcCovarMatrix(X1, Covar1, Mu1, CV_COVAR_NORMAL+CV_COVAR_ROWS, CV_64F);

  cv::Mat XM0(X0.size(), X0.type()), XM1(X1.size(), X1.type());
  for(int i = 0; i < X0.rows; i++) {
    for(int j = 0; j < dimensions; j++) {
      XM0.at<double>(i, j) = X0.at<double>(i, j) - Mu0.at<double>(j);
    }
  }

  for(int i = 0; i < X1.rows; i++) {
    for(int j = 0; j < dimensions; j++) {
      XM1.at<double>(i, j) = X1.at<double>(i, j) - Mu1.at<double>(j);
    }
  }

  Sw = (XM0.t() * XM0) + (XM1.t() * XM1);
  
  cv::Mat M0M(dimensions, 1, CV_64F);
  for(int j = 0; j < dimensions; j++) {
    M0M.at<double>(j) = Mu0.at<double>(j) - Mu.at<double>(j);
  }

  cv::Mat M1M(dimensions, 1, CV_64F);
  for(int j = 0; j < dimensions; j++) {
    M1M.at<double>(j) = Mu1.at<double>(j) - Mu.at<double>(j);
  }

  Sb = n0 * M0M * M0M.t() + n1 * M1M * M1M.t();

  cv::Mat S = Sw.inv(cv::DECOMP_SVD) * Sb;

  cv::Mat eigenvalues(1, dimensions, Sb.type()), eigenvectors(dimensions, dimensions, Sb.type());
  cv::eigen(S, eigenvalues, eigenvectors);

  cv::Mat W;

  W.push_back(eigenvectors.row(0));
  W.push_back(eigenvectors.row(1));
  W.push_back(eigenvectors.row(2));

  return W;
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

  cv::Mat samples(bgImage.cols*bgImage.rows+skinImage.cols*skinImage.rows, 6, CV_64F);
  std::vector<int> labels(bgImage.cols*bgImage.rows+skinImage.cols*skinImage.rows);

  cv::Mat bgHSVImage, skinHSVImage;

  cv::cvtColor( bgImage, bgHSVImage, CV_BGR2HSV );
  cv::cvtColor( skinImage, skinHSVImage, CV_BGR2HSV );

  bgImage.convertTo(bgImage, CV_64F);
  skinImage.convertTo(skinImage, CV_64F);
  bgHSVImage.convertTo(bgHSVImage, CV_64F);
  skinHSVImage.convertTo(skinHSVImage, CV_64F);


  int count = 0;

  for( int y = 0; y < bgImage.rows ; y++ ) { 
    for( int x = 0; x < bgImage.cols; x++ ) {
      cv::Vec3d color = bgImage.at<cv::Vec3d>(y, x);
      cv::Vec3d colorHSV = bgHSVImage.at<cv::Vec3d>(y, x);

      samples.at<double>(count, 0) = color[0];
      samples.at<double>(count, 1) = color[1];
      samples.at<double>(count, 2) = color[2];

      samples.at<double>(count, 3) = colorHSV[0];
      samples.at<double>(count, 4) = colorHSV[1];
      samples.at<double>(count, 5) = colorHSV[2];

      labels[count] = 0;

      count++;
    }
  }

  for( int y = 0; y < skinImage.rows ; y++ ) { 
    for( int x = 0; x < skinImage.cols; x++ ) {
      cv::Vec3d color = skinImage.at<cv::Vec3d>(y, x);
      cv::Vec3d colorHSV = skinHSVImage.at<cv::Vec3d>(y, x);

      samples.at<double>(count, 0) = color[0];
      samples.at<double>(count, 1) = color[1];
      samples.at<double>(count, 2) = color[2];

      samples.at<double>(count, 3) = colorHSV[0];
      samples.at<double>(count, 4) = colorHSV[1];
      samples.at<double>(count, 5) = colorHSV[2];

      labels[count] = 1;

      count++;
    }
  }

  std::ofstream ofs, ofs2;
  ofs.open( "X.txt" );
  ofs2.open( "y.txt" );

  for( int i = 0; i < samples.rows; i++) {
    ofs << samples.at<double>(i, 0) << " " << samples.at<double>(i, 1) << " ";
    ofs << samples.at<double>(i, 2) << " " << samples.at<double>(i, 3) << " ";
    ofs << samples.at<double>(i, 4) << " " << samples.at<double>(i, 5) << std::endl;

    ofs2 << labels[i] << std::endl;
  }

  ofs.close(); ofs2.close();

  cv::Mat W = lda(samples, labels);

  cv::Mat imageHSV;

  cv::cvtColor(image, imageHSV, CV_BGR2HSV);
  image.convertTo(image, CV_64F);
  imageHSV.convertTo(imageHSV, CV_64F);
  cv::imwrite("cbcr.jpg", image);

  count = 0;
  for( int y = 0; y < image.rows ; y++ ) { 
    for( int x = 0; x < image.cols; x++ ) {
      cv::Vec3d color = image.at<cv::Vec3d>(y, x);
      cv::Vec3d colorHSV = imageHSV.at<cv::Vec3d>(y, x);

      cv::Mat sample(6, 1, CV_64F), resample(3, 1, CV_64F);

      sample.at<double>(0) = color[0];
      sample.at<double>(1) = color[1];
      sample.at<double>(2) = color[2];

      sample.at<double>(3) = colorHSV[0];
      sample.at<double>(4) = colorHSV[1];
      sample.at<double>(5) = colorHSV[2];

      resample = W * sample;

      image.at<cv::Vec3d>(y, x) = cv::Vec3d(resample.at<double>(0), resample.at<double>(1), resample.at<double>(2));

      count++;
    }
  }

  cv::imwrite("cbcr2.jpg", image);

  return EXIT_SUCCESS;
}

