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
	cv::ximgproc::getDisparityVis(disp_map, result);
	imshow("disparity map", result);
	waitKey(0);
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

	displayDispMap(disp_map);

	return disp_map;
}

