#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>

#include "Eigen.h"
#include "common_types.h"
#include "rectify.h"
#include "disparity.h"


#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
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
Mat computeDepthMap(Mat disp_map, double baseline, Mat cam_m, double doffs);
void writeDepthMap(Mat depth_map);
void display_gt(double baseline, double doffs, Mat calib);

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

	//Camera parameters
	cv::Mat cameraMatrix0 = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat cameraMatrix1 = cv::Mat::zeros(3, 3, CV_64F);
	double doffs; double baseline;
	int width; int height;
	int ndisp; int isint; int vmin; int vmax;
	double dyavg; double dymax;

	std::string cameraParamDir = "../data/Motorcycle-perfect/calib.txt";//Change this line to the directory of the txt file
	readCameraCalib(cameraParamDir, cameraMatrix0, cameraMatrix1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax);
	//display_gt(baseline, doffs, cameraMatrix0);
	//return 0;
	
	//Rectify images shoud work without any issue 
	// rectifyImages(imgL, imgR, rectL, rectR, cameraMatrix0, cameraMatrix1, baseline, width, height);
	//visualization of rectified images
	// visualizeRectified(rectL, rectR, width, height);
	Mat disp_map = computeDisparityMap(imgL, imgR);
	Mat _3DImage = computeDepthMap(disp_map, baseline, cameraMatrix0, doffs);
	writeDepthMap(_3DImage);
	return 0;
	
}


void display_gt(double baseline, double doffs, Mat calib_cam)
{
	std::stringstream ss;
	ss << dataset_path << "/disp0.pfm";
	Mat img_gt = imread(ss.str(), IMREAD_UNCHANGED);
	displayDispMap(img_gt);
	Mat depth_map = computeDepthMap(img_gt, baseline, calib_cam, doffs);
	writeDepthMap(depth_map);
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

	Mat imgr = imread(ssr.str(), IMREAD_GRAYSCALE);
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

Mat computeDepthMap(Mat disp_map, double baseline, Mat cam_m, double doffs)
{
	Mat depthMap;

	cv::Mat Q(4,4, CV_64F);
	Q.at<double>(0, 0) = 1.0;
	Q.at<double>(0, 1) = 0.0;
	Q.at<double>(0, 2) = 0.0;
	Q.at<double>(0, 3) = -cam_m.at<double>(0,2); //cx
	Q.at<double>(1, 0) = 0.0;
	Q.at<double>(1, 1) = 1.0;
	Q.at<double>(1, 2) = 0.0;
	Q.at<double>(1, 3) = -cam_m.at<double>(1,2);  //cy
	Q.at<double>(2, 0) = 0.0;
	Q.at<double>(2, 1) = 0.0;
	Q.at<double>(2, 2) = 0.0;
	Q.at<double>(2, 3) = cam_m.at<double>(0,0);  //Focal
	Q.at<double>(3, 0) = 0.0;
	Q.at<double>(3, 1) = 0.0;
	Q.at<double>(3, 2) = 1.0 / baseline;    //1.0/BaseLine
	Q.at<double>(3, 3) = doffs;    //cx - cx'
	
	Mat floatDisp;
	if (disp_map.type() == CV_16S)
	{
		disp_map.convertTo(floatDisp, CV_32F, 1.0f / 16.0);
	}
	
	reprojectImageTo3D(floatDisp, depthMap, Q);

	return depthMap;
}

void writeDepthMap(Mat depthMap)
{
	int valid = 0;
	for (int i = 0; i < depthMap.rows; i++)
		for (int j = 0; j < depthMap.cols; j++)
		{
			Vec3f point = depthMap.at<Vec3f>(i, j);
			if (!isnan(point[0]))
			{
				valid++;
			}
		}
	std::ofstream outFile("moto.ply");
	outFile << "ply" << std::endl;
	outFile << "format ascii 1.0" << std::endl;
	outFile << "element vertex " << valid << std::endl;
	outFile << "property float x" << std::endl;
	outFile << "property float y" << std::endl;
	outFile << "property float z" << std::endl;
	outFile << "end_header" << std::endl;

	for(int i=0;i<depthMap.rows;i++)
		for (int j = 0; j < depthMap.cols; j++)
		{
			Vec3f point = depthMap.at<Vec3f>(i, j);
			if (!isnan(point[0]))
			{
				outFile << point[0] << " " << point[1] << " " << point[2] << std::endl;
			}
		}
	outFile.close();
}