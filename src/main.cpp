#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <filesystem>

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

#include <chrono>

using namespace cv;
using namespace stereo_vis;

namespace fs = std::filesystem;

// Declarations

void read_images();
void display_images();
Mat computeDepthMap(Mat disp_map, double baseline, Mat cam_m, double doffs);
void writeDepthMap(Mat depth_map, Mat colors);
void display_gt(double baseline, double doffs, Mat calib);
void run_evaluation();

// Constants

const std::string dataset_path = "../data/Middlebury/Motorcycle-perfect";

// Variables

std::unordered_map<FrameCamId, cv::Mat> images;
const std::string matcher_type("SGBM"); // BM or SGBM or ORB

int main() 
{
	//run_evaluation();
	//return 0;
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

	std::string cameraParamDir = "../data/Middlebury/Motorcycle-perfect/calib.txt";//Change this line to the directory of the txt file
	readCameraCalib(cameraParamDir, cameraMatrix0, cameraMatrix1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax);

	// display_gt(baseline, doffs, cameraMatrix0);
	
	AbstractStereoMatcher* matcher = AbstractStereoMatcher::Create(matcher_type);

	// rectifyImages(imgL, imgR, rectL, rectR, cameraMatrix0, cameraMatrix1, baseline, width, height);

	Mat disp_map = matcher->computeDisparityMap(imgL, imgR);
	displayDispMap(disp_map);
	Mat _3DImage = computeDepthMap(disp_map, baseline, cameraMatrix0, doffs);

	Mat colors;
	cvtColor(img_color, colors, COLOR_BGR2RGB);
	writeDepthMap(_3DImage, colors);
	return 0;
	
}

void display_gt(double baseline, double doffs, Mat calib_cam)
{
	std::stringstream ss;
	ss << dataset_path << "/disp0.pfm";
	Mat img_gt = imread(ss.str(), IMREAD_UNCHANGED);
	img_gt.setTo(0, img_gt == INFINITY);
	displayDispMap(img_gt);
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


void run_evaluation()
{
	const std::string dataset_folder = "../data/Middlebury";
	const std::string matcher_type = "SGBM";
	int frame_id = 0;

	// Camera parameters
	cv::Mat cameraMatrix0 = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat cameraMatrix1 = cv::Mat::zeros(3, 3, CV_64F);
	double doffs; double baseline;
	int width; int height;
	int ndisp; int isint; int vmin; int vmax;
	double dyavg; double dymax;

	float ratios_total = 0;
	float n_ratios = 0;

	double avg_execution_time = 0;

	for (const auto& entry : fs::directory_iterator(dataset_folder))
	{
		std::cout << "Dataset: " << entry.path().filename() << std::endl;
		FrameCamId fcidl(frame_id, 0);
		FrameCamId fcidr(frame_id, 1);

		std::stringstream ssl;
		ssl << entry.path().string() << "/" << "im0.png";

		std::stringstream ssr;
		ssr << entry.path().string() << "/" << "im1.png";

		Mat imgl = imread(ssl.str(), IMREAD_GRAYSCALE);
		Mat imgr = imread(ssr.str(), IMREAD_GRAYSCALE);
		Mat img_color = imread(ssl.str(), IMREAD_COLOR);

		images[fcidl] = imgl;
		images[fcidr] = imgr;

		// read ground-truth disparity map
		std::stringstream ss;
		ss << entry.path().string() << "/disp0.pfm";
		Mat img_gt = imread(ss.str(), IMREAD_UNCHANGED);
		img_gt.setTo(0, img_gt == INFINITY);
		// displayDispMap(img_gt);
		normalize(img_gt, img_gt, 0, 256, NORM_MINMAX, CV_8U);

		// read camera parameters
		std::stringstream cameraParamDir;
		cameraParamDir << entry.path().string() << "/calib.txt";
		readCameraCalib(cameraParamDir.str(), cameraMatrix0, cameraMatrix1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax);

		AbstractStereoMatcher* matcher = AbstractStereoMatcher::Create(matcher_type);
		
		auto start = std::chrono::high_resolution_clock::now();
		Mat disp_map = matcher->computeDisparityMap(imgl, imgr);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		std::cout << "Duration: " << duration.count() << std::endl;
		avg_execution_time += duration.count();
		Mat disp_map_normalized;
		normalize(disp_map, disp_map_normalized, 0, 256, NORM_MINMAX, CV_8U);

		int width_offset = (int)(disp_map.cols * 0.1); // offset of 10% to avoid the black band on the left

		int outliers = 0;
		int outliers_non_occluded = 0;
		int totalPix_non_occluded = 0;
		for(int i=0; i < disp_map.rows; i++)
			for (int j = width_offset; j < disp_map.cols; j++)
			{
				int d_computed = disp_map_normalized.at<uchar>(i, j);
				int d_gt = img_gt.at<uchar>(i, j);
				int pix_err = abs(d_computed - d_gt);
				if (pix_err > 30)
				{
					outliers++;
				}
				if (pix_err > 10 && d_computed != 0 && d_gt != 0)
				{
					outliers_non_occluded++;
				}
				if (d_computed == 0 || d_gt == 0)
				{
					totalPix_non_occluded++;
				}
			}

		int totalPix = disp_map.rows * (disp_map.cols - width_offset);

		float ratio = (float)outliers * 100 / totalPix;
		float ratio_non_occluded = (float)outliers_non_occluded * 100 / (totalPix - totalPix_non_occluded);

		std::cout << "Ratio: " << ratio << std::endl;
		std::cout << "Ratio non-occluded: " << ratio_non_occluded << std::endl << std::endl;

		if (ratio < 60)
		{
			ratios_total += ratio;
			n_ratios++;
		}

		frame_id++;
	}
	std::cout << "Ratio: " << ratios_total / n_ratios << std::endl;
	std::cout << "Number of ratios: " << n_ratios << std::endl;
	std::cout << "Average execution time: " << avg_execution_time / frame_id << std::endl;
}