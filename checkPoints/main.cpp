#include <iostream>
#include <sstream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

int readMolePoints(const std::string & molePointsFilePath,
                    std::vector<cv::KeyPoint> & fixedPoints,
                    const int imgWidth, const int imgHeight)
{
  std::string line;
  std::ifstream molePointsFile (molePointsFilePath.c_str());

  if(!molePointsFile.is_open()) {
    std::cerr << "Unable to open input file " << molePointsFilePath << std::endl;
    return EXIT_FAILURE;
  }

  while(molePointsFile.good()) {
    getline(molePointsFile, line);
    std::istringstream buffer(line); int x,y;
    buffer >> x; buffer >> y; buffer.clear();
    
    if(molePointsFile.eof()){ break; }

    if(x < 0 || x >= imgWidth || y < 0 || y >= imgHeight) {
      std::cerr << "Invalid mole point coordinate (" << x << ", " << y << ")" << std::endl;
      return EXIT_FAILURE;
    }

    fixedPoints.push_back(cv::KeyPoint(x, y, 1));
  }

  return EXIT_SUCCESS;
}

int main( int argc, char** argv )
{
  if( argc != 3 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <PointsFile>" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat image = cv::imread( argv[1], 1 );
  if(!image.data) {
    return EXIT_FAILURE;
  }

  std::vector<cv::KeyPoint> points;
  readMolePoints(argv[2], points, image.cols, image.rows);

  cv::Mat imagePoints;
  cv::drawKeypoints(image, points, imagePoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

  char* source_window = "Source image";
  cv::namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  cv::imshow( source_window, imagePoints );

  cv::waitKey(0);
  return EXIT_SUCCESS;
}