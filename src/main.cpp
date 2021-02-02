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
Mat computeDisparityMap(Mat rect1, Mat rect2);
void displayDispMap(Mat disp_map);
Mat filterDispMap(Mat rect1, Mat rect2);

// Constants

//const std::string dataset_path = "/home/abhishek/Stereo_Reconstruction/data/Motorcycle-perfect";
const std::string dataset_path = "../data/Motorcycle-perfect";

// Variables

std::unordered_map<FrameCamId, cv::Mat> images;


int main() 
{
	read_images();
//	 display_images();
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
//
	std::string cameraParamDir = "../data/Motorcycle-perfect/calib.txt";//Change this line to the directory of the txt file
	readCameraCalib(cameraParamDir, cameraMatrix0, cameraMatrix1, doffs, baseline, width, height, ndisp, isint, vmin, vmax, dyavg, dymax);
//
	//Rectify images shoud work without any issue
	Mat Q; //for disparity to depth conversion
	Q = rectifyImages(imgL, imgR, rectL, rectR, cameraMatrix0, cameraMatrix1, baseline, width, height);
//	//visualization of rectified images
//	visualizeRectified(rectL, rectR, width, height);
	Mat disp_map = computeDisparityMap(rectL, rectR);
//    disp_map = filterDispMap(rectL, rectR);

    Mat points;
    Mat points1;
//    reprojectImageTo3D(disp_map/16.0, points, Q, true);
    disp_map.convertTo(disp_map,CV_32F);
    reprojectImageTo3D(disp_map/16.0, points, Q, false, CV_32F);

    std::ofstream point_cloud_file;
    point_cloud_file.open ("point_cloud.xyz");
    for(int i = 0; i < points.rows; i++) {
        for(int j = 0; j < points.cols; j++) {
            Vec3f point = points.at<Vec3f>(i,j);
             {
                point_cloud_file << point[0] << " " << point[1] << " " << point[2]
                                 << " " << static_cast<unsigned>(rectL.at<uchar>(i,j)) << " " << static_cast<unsigned>(rectL.at<uchar>(i,j)) << " " << static_cast<unsigned>(rectL.at<uchar>(i,j)) << std::endl;
            }
        }
    }
    point_cloud_file.close();

	return 0;
}

void read_images()
{
	FrameCamId fcidl(0, 0);
	FrameCamId fcidr(0, 1);

	std::stringstream ssl;
	ssl << dataset_path << "/" << "im0" << ".png";

	Mat imgl = imread(ssl.str(), 0);
	images[fcidl] = imgl;

	std::stringstream ssr;
	ssr << dataset_path << "/" << "im1" << ".png";

	Mat imgr = imread(ssr.str(), 0);
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
