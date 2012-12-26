#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

int main( int argc, char** argv )
{
  cv::Mat image, gray_image;
  image = cv::imread( argv[1], 1 );

  if( argc != 2 || !image.data ) {
    printf( "No image data \n" );
    return -1;
  }

  cv::namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Display Image", gray_image );

  cv::waitKey(0);

  return 0;
}
