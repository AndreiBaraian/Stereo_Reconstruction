#include <iostream>
#include <fstream>

#include "Eigen.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;


int main() 
{
	Mat img = imread("../data/test_img.png", IMREAD_COLOR);
	imshow("test_img", img);
	waitKey(0);
	return 0;
}
