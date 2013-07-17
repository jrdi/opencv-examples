#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "eigen3/Eigen/Eigenvalues"
#include "opencv2/core/eigen.hpp"

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

  // cv::Mat W;

  // W.push_back(eigenvectors.row(0));
  // W.push_back(eigenvectors.row(1));
  // W.push_back(eigenvectors.row(2));

  Eigen::MatrixXd A( 6, 6 );
  cv::cv2eigen(S, A);

  Eigen::EigenSolver<Eigen::MatrixXd> es(A);
  Eigen::VectorXd eigenval = es.eigenvalues().real();
  Eigen::MatrixXd eigenvect = es.eigenvectors().real();

  typedef std::multimap< double, Eigen::VectorXd, std::greater< double > >    EigenSorterType;
  EigenSorterType D;

  for (int i=0;i<eigenval.size();i++) {
    D.insert( EigenSorterType::value_type( std::abs(eigenval.coeff(i, 0)), eigenvect.col(i) ) );
  }
  
  Eigen::MatrixXd sortedEigs;
  sortedEigs.resizeLike(eigenvect);
  int i = 0;
  for (EigenSorterType::const_iterator it = D.begin(); it != D.end(); ++it) {
    sortedEigs.col(i) = it->second;
    i++;
  }
  eigenvect = sortedEigs;

  Eigen::MatrixXd WEig( 6, 3 );

  WEig.col(0) = eigenvect.col(0);
  WEig.col(1) = eigenvect.col(1);
  WEig.col(2) = eigenvect.col(2);

  std::cout << "The eigenvalues of W are:" << std::endl << WEig << std::endl;
  std::cout << "The matrix of eigenvectors, V, is:" << std::endl << eigenvect << std::endl << std::endl;

  cv::Mat W(3, 6, CV_64F, WEig.transpose().data());

  std::cout << W << std::endl;

  return W;
}

int main( int argc, char** argv )
{
  if( argc != 3 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <mask>" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat image = cv::imread( argv[1], 1 );
  if(!image.data) {
    return EXIT_FAILURE;
  }
  cv::Mat mask = cv::imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
  if(!mask.data) {
    return EXIT_FAILURE;
  }

  cv::Mat samples(image.rows*image.cols, 6, CV_64F);
  std::vector<int> labels(image.rows*image.cols);

  cv::Mat imageHSV;

  cv::cvtColor( image, imageHSV, CV_BGR2HSV );

  image.convertTo(image, CV_64F);
  imageHSV.convertTo(imageHSV, CV_64F);

  int count = 0;

  for( int y = 0; y < image.rows ; y++ ) { 
    for( int x = 0; x < image.cols; x++ ) {
      cv::Vec3d color = image.at<cv::Vec3d>(y, x);
      cv::Vec3d colorHSV = imageHSV.at<cv::Vec3d>(y, x);

      samples.at<double>(count, 0) = color[0];
      samples.at<double>(count, 1) = color[1];
      samples.at<double>(count, 2) = color[2];

      samples.at<double>(count, 3) = colorHSV[0];
      samples.at<double>(count, 4) = colorHSV[1];
      samples.at<double>(count, 5) = colorHSV[2];

      if(mask.at<uchar>(y, x) > 120) {
        labels[count] = 1;
      } else {
        labels[count] = 0;
      }

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

  cv::imwrite("original.jpg", image);

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

  cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);

  std::vector<cv::Mat> channels;
  cv::split(image, channels);

  std::ofstream myFile;
  myFile.open( "img0.csv" );
  
  std::stringstream ss;
  ss << format(channels[0],"csv") << std::endl << std::endl;
  myFile << ss.str();
  
  myFile.close();

  myFile.open( "img1.csv" );
  
  ss << format(channels[1],"csv") << std::endl << std::endl;
  myFile << ss.str();
  
  myFile.close();

  myFile.open( "img2.csv" );
  
  ss << format(channels[2],"csv") << std::endl << std::endl;
  myFile << ss.str();
  
  myFile.close();

  cv::imwrite("projected0.jpg", channels[0]);
  cv::imwrite("projected1.jpg", channels[1]);
  cv::imwrite("projected2.jpg", channels[2]);

  cv::imwrite("projected.jpg", image);

  return EXIT_SUCCESS;
}

