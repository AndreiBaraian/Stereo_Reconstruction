#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

using namespace cv;

void displayDispMap(Mat disp_map)
{
	Mat result;
	double w = disp_map.cols;
	double h = disp_map.cols;
	double ratio = 600. / std::max(w, h);
	w = cvRound(w * ratio);
	h = cvRound(h * ratio);
	namedWindow("disparity map", WINDOW_NORMAL);
	resizeWindow("disparity map", w, h);
	
	/*
	if (disp_map.type() == CV_16S)
	{
		disp_map.convertTo(disp_map, CV_32F, 1.0 / 16.0, 0.0);
	}
	*/

	normalize(disp_map, result, 0, 256, NORM_MINMAX, CV_8U);
	// result.convertTo(temp, CV_8U);
	// cv::ximgproc::getDisparityVis(disp_map, result);
	imshow("disparity map", result);
	waitKey(0);
}

Mat computeDisparityMap(Mat rect1, Mat rect2)
{
	Mat disp_map;

	int ndisparities = 288;
	int SADWindowSize = 15;

	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

	//sbm->setP1(8 * cn * SADWindowSize * SADWindowSize);
	//sbm->setP2(32 * cn * SADWindowSize * SADWindowSize);

	//sbm->setMode(StereoSGBM::MODE_HH);
	

	sbm->compute(rect1, rect2, disp_map);

	displayDispMap(disp_map);

	return disp_map;
}

