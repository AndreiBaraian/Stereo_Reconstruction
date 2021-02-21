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
void writeDepthMap(Mat depth_map, Mat colors);
void display_gt(double baseline, double doffs, Mat calib);

// Constants

const std::string dataset_path = "../data/Motorcycle-perfect";

// Variables

std::unordered_map<FrameCamId, cv::Mat> images;
const std::string current_mode ("ORB");//currently implemented modes are "SGBM", "BM" and, "groundtruth", "ORB"

int main() 
{
	read_images();
	// display_images();
	FrameCamId fcidl(0, 0);
	FrameCamId fcidr(0, 1);
	Mat imgL = images[fcidl];
	Mat imgR = images[fcidr];
	Mat rectL, rectR;

   //to get the colored image for colored mesh
    std::stringstream ssl;
    ssl << dataset_path << "/" << "im0" << ".png";
    Mat img_color = imread(ssl.str(), IMREAD_COLOR);

	//Camera parameters
	cv::Mat cameraMatrix0 = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat cameraMatrix1 = cv::Mat::zeros(3, 3, CV_64F);
	double doffs; double baseline;
	int width; int height;
	int ndisp; int isint; int vmin; int vmax;
	double dyavg; double dymax;

	std::string cameraParamDir = "../data/Motorcycle-perfect/calib.txt";//Change this line to the directory of the txt file
	readCameraCalib(cameraParamDir, cameraMatrix0, cameraMatrix1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax);

	if(current_mode.compare("groundtruth") == 0)
	{
		std::cout << "hello" << std::endl;
		display_gt(baseline, doffs, cameraMatrix0);
		return 0;
	}
	else if (current_mode.compare("BM") == 0)
	{
		//Rectify images shoud work without any issue 
		// rectifyImages(imgL, imgR, rectL, rectR, cameraMatrix0, cameraMatrix1, baseline, width, height);
		//visualization of rectified images
		// visualizeRectified(rectL, rectR, width, height);
		//Mat disp_map = computeDisparityMap(imgL, imgR);
		Mat disp_map = computeDisparityMapBM(imgL, imgR);


		Mat _3DImage = computeDepthMap(disp_map, baseline, cameraMatrix0, doffs);
		Mat colors;
		cvtColor(img_color, colors, COLOR_BGR2RGB);
//		std::cout<<"Colors : "<< colors.at<Vec3b>(0,0)  <<std::endl;
		writeDepthMap(_3DImage, colors);
		return 0;
	}

	else if (current_mode.compare("SGBM") == 0)
	{
		//Rectify images shoud work without any issue 
		rectifyImages(imgL, imgR, rectL, rectR, cameraMatrix0, cameraMatrix1, baseline, width, height);
		//visualization of rectified images
		// visualizeRectified(rectL, rectR, width, height);
		//Mat disp_map = computeDisparityMap(imgL, imgR);
		Mat disp_map = computeDisparityMapSGBM(imgL, imgR);
		Mat _3DImage = computeDepthMap(disp_map, baseline, cameraMatrix0, doffs);
		Mat colors;
        cvtColor(img_color, colors, COLOR_BGR2RGB);
		writeDepthMap(_3DImage, colors);
		return 0;
	}

	else if (current_mode.compare("ORB") == 0)
	{
		//Rectify images shoud work without any issue 
		rectifyImages(imgL, imgR, rectL, rectR, cameraMatrix0, cameraMatrix1, baseline, width, height);
		//visualization of rectified images
		//visualizeRectified(rectL, rectR, width, height);
		//Mat disp_map = computeDisparityMap(imgL, imgR);
		Mat disp_map = computeDisparityMapORB(rectL, rectR, 400, 0);
		imwrite("test_dips.png", disp_map);
		Mat _3DImage = computeDepthMap(disp_map, baseline, cameraMatrix0, doffs);
		Mat colors;
		cvtColor(img_color, colors, COLOR_BGR2RGB);
		writeDepthMap(_3DImage, colors);
		return 0;
	}
}

void display_gt(double baseline, double doffs, Mat calib_cam)
{
	std::stringstream ss;
	ss << dataset_path << "/disp0.pfm";
	Mat img_gt = imread(ss.str(), IMREAD_UNCHANGED);
	img_gt.setTo(0, img_gt == INFINITY);
	// displayDispMap(img_gt);
	img_gt.setTo(INFINITY, img_gt == 0);
	// normalize(img_gt, img_gt, 0, 256, NORM_MINMAX, CV_8U);
	Mat depth_map = computeDepthMap(img_gt, baseline, calib_cam, doffs);
	Mat colors;
    cvtColor(img_gt, colors, COLOR_BGR2RGB);
	writeDepthMap(depth_map, colors);
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
	
	Q.convertTo(Q, CV_32F);

	if (disp_map.type() == CV_16S)
	{
		disp_map.convertTo(disp_map, CV_32F, 16.0);
		disp_map.setTo(INFINITY, disp_map < 0);
	}

	reprojectImageTo3D(disp_map, depthMap, Q, false, CV_32F);
	
	return depthMap;
}

void writeDepthMap(Mat depthMap, Mat colors)
{
	int valid = 0;
	for (int i = 0; i < depthMap.rows; i++)
		for (int j = 0; j < depthMap.cols; j++)
		{
			Vec3f point = depthMap.at<Vec3f>(i, j);
			if (!isnan(point[0]) && point[2] > 5)
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
	outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
	outFile << "end_header" << std::endl;

	for(int i=0;i<depthMap.rows;i++)
		for (int j = 0; j < depthMap.cols; j++)
		{
			Vec3f point = depthMap.at<Vec3f>(i, j);
			Vec3f color = colors.at<Vec3b>(i, j);
			if (!isnan(point[0]) && point[2] > 5)
			{
				outFile << point[0] << " " << point[1] << " " << point[2] << " " << color[0] << " " << color[1] << " " << color[2] << std::endl;
			}
		}
	outFile.close();
}