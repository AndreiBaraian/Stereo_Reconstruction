#include <iostream>
#include <fstream>
#include <unordered_map>

#include "Eigen.h"
#include "common_types.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace stereo_vis;

// Declarations

void read_images();
void display_images();

// Constants

const std::string dataset_path = "../data/Backpack-perfect";

// Variables

std::unordered_map<FrameCamId, cv::Mat> images;


int main() 
{
	read_images();
	display_images();
	return 0;
}

void read_images()
{
	FrameCamId fcidl(0, 0);
	FrameCamId fcidr(0, 1);

	std::stringstream ssl;
	ssl << dataset_path << "/" << "im0" << ".png";

	Mat imgl = imread(ssl.str(), IMREAD_GRAYSCALE);
	images[fcidl] = imgl;

	std::stringstream ssr;
	ssr << dataset_path << "/" << "im1" << ".png";

	Mat imgr = imread(ssl.str(), IMREAD_GRAYSCALE);
	images[fcidr] = imgr;
}

void display_images()
{
	for (auto const kv : images)
	{
		std::stringstream ss;
		ss << kv.first.cam_id;
		namedWindow(ss.str(), WINDOW_NORMAL);
		resizeWindow(ss.str(), 600, 600);
		imshow(ss.str(), kv.second);
	}
	waitKey(0);
}