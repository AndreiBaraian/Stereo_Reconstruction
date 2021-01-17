#include <iostream>
#include <fstream>
#include <unordered_map>

#include "Eigen.h"
#include "common_types.h"
#include "rectify.h"


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>



using namespace cv;
using namespace stereo_vis;

// Declarations

void read_images();
void display_images();
Mat computeDisparityMap(Mat rect1, Mat rect2);
void displayDispMap(Mat disp_map);
void rectifyImages(Mat imgL, Mat imgR, Mat& rectL, Mat& rectR);
void readCalibrationMiddleburry(Mat &P1, Mat&P2, int& ndisp);

// Constants

const std::string dataset_path = "../data/Motorcycle-perfect";

// Variables

std::unordered_map<FrameCamId, cv::Mat> images;


int main() 
{
	read_images();
	// display_images();
	FrameCamId fcidl(0, 0);
	FrameCamId fcidr(0, 1);
	Mat imgL = images[fcidl];
	Mat imgR = images[fcidr];
	Mat rectL, rectR;
	//rectifyImages(imgL, imgR, rectL, rectR);

	//Camera parameters
	cv::Mat cameraMatrix0 = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat cameraMatrix1 = cv::Mat::zeros(3, 3, CV_64F);
	double doffs; double baseline;
	int width; int height;
	int ndisp; int isint; int vmin; int vmax;
	double dyavg; double dymax;

	std::string cameraParamDir = "Motorcycle-imperfect/calib.txt";//Change this line to the directory of the txt file
	readCameraCalib(cameraParamDir, cameraMatrix0, cameraMatrix1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax);

	//Rectify images shoud work without any issue 
	rectifyImages(imgL, imgR, rectL, rectR, cameraMatrix0, cameraMatrix1, baseline, width, height);
	//visualization of rectified images
	visualizeRectified(rectL, rectR, width, height);

	Mat disp_map = computeDisparityMap(rectL, rectR);
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

void readCalibrationMiddleburry(Mat &P1, Mat&P2, int& ndisp)
{
	// not finished
	std::stringstream calibration_file;
	calibration_file << dataset_path << "/calib.txt";

	std::ifstream f;
	f.open(calibration_file.str());
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


void displayDispMap(Mat disp_map)
{
	double minVal, maxVal;
	minMaxLoc(disp_map, &minVal, &maxVal);
	
	Mat disp_map_8U = Mat(disp_map.rows, disp_map.cols, CV_8UC1);
	disp_map.convertTo(disp_map_8U, CV_8UC1, 255 / (maxVal - minVal));

	//namedWindow("disparity map", WINDOW_NORMAL);
	//resizeWindow("disparity map", 600, 600);
	imshow("disparity map", disp_map_8U);
	waitKey(0);
}

void rectifyImages(Mat imgL, Mat imgR, Mat& rectL, Mat& rectR)
{
	Mat P1 = (Mat_<double>(3, 3) << 3979.911, 0, 1244.772, 0, 3979.911, 1019.507, 0, 0, 1);
	Mat P2 = (Mat_<double>(3, 3) << 3979.911, 0, 1369.115, 0, 3979.911, 1019.507, 0, 0, 1);

	Mat D;
	Mat R;
	Mat map1x, map1y, map2x, map2y;

	initUndistortRectifyMap(P1, D, R, P1, imgL.size(), CV_32FC1, map1x, map1y);
	remap(imgL, rectL, map1x, map1y, INTER_LINEAR);

	initUndistortRectifyMap(P2, D, R, P2, imgR.size(), CV_32FC1, map2x, map2y);
	remap(imgR, rectR, map2x, map2y, INTER_LINEAR);
}

Mat computeDisparityMap(Mat rect1, Mat rect2)
{
	Mat disp_map = Mat(rect1.rows, rect2.cols, CV_16S);

	int ndisparities = 256;
	int SADWindowSize = 15;

	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	/*
	sbm->setPreFilterCap(32);
	sbm->setBlockSize(SADWindowSize);
	int cn = rect1.channels();
	sbm->setP1(8 * cn * SADWindowSize * SADWindowSize);
	sbm->setP2(32 * cn * SADWindowSize * SADWindowSize);
	sbm->setMinDisparity(0);
	sbm->setNumDisparities(128);
	sbm->setUniquenessRatio(10);
	sbm->setSpeckleWindowSize(100);
	sbm->setSpeckleRange(32);
	sbm->setDisp12MaxDiff(1);
	sbm->setMode(StereoSGBM::MODE_HH);
	*/
	sbm->compute(rect1, rect2, disp_map);

	//displayDispMap(disp_map);
	double minVal, maxVal;
	minMaxLoc(disp_map, &minVal, &maxVal);

	Mat disp_map_8U = Mat(disp_map.rows, disp_map.cols, CV_8UC1);
	disp_map.convertTo(disp_map_8U, CV_8UC1, 255 / (maxVal - minVal));

	namedWindow("disparity map", WINDOW_NORMAL);
	resizeWindow("disparity map", 600, 600);
	Mat result;
	cv::ximgproc::getDisparityVis(disp_map, result);
	imshow("disparity map", disp_map_8U);
	waitKey(0);
	return disp_map;
}