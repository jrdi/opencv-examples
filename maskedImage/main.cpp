#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

int main( int argc, char** argv )
{
  if( argc != 4 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <roiImage> <OutputImage>" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat image = cv::imread( argv[1], 1 );
  cv::Mat roi = cv::imread( argv[2], 1 );
  if(!image.data) {
    return EXIT_FAILURE;
  }

  cv::Mat imgROI;
  cv::bitwise_and(image, roi, imgROI);

  cv::imwrite(argv[3], imgROI);

  return EXIT_SUCCESS;
}
