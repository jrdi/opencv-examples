#include <iostream>

#include "cv.h"
#include "highgui.h"

int main( int argc, char** argv )
{
  if( argc != 2 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage>" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat image = cv::imread( argv[1], 1 );
  if(!image.data) {
    return EXIT_FAILURE;
  }

  cv::namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Display Image", image );
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
