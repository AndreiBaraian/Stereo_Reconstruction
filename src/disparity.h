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
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

void displayDispMap(Mat disp_map)
{
	Mat result;
	double w = disp_map.cols;
	double h = disp_map.rows;
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

Mat computeDisparityMapBM(Mat rect1, Mat rect2)
{
	Mat disp_map;

	int ndisparities = 288;
	int SADWindowSize = 21;

	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

	//sbm->setP1(8 * cn * SADWindowSize * SADWindowSize);
	//sbm->setP2(32 * cn * SADWindowSize * SADWindowSize);

	sbm->setSpeckleWindowSize(200);
	sbm->setSpeckleRange(64);
	sbm->setTextureThreshold(10);

	sbm->compute(rect1, rect2, disp_map);

	displayDispMap(disp_map);

	return disp_map;
}

Mat computeDisparityMapSGBM(Mat rect1, Mat rect2)
{
	Mat left_disp;

	int ndisparities = 15;
	int SADWindowSize = 250;

	Ptr<StereoSGBM> left_matcher = StereoSGBM::create(ndisparities, SADWindowSize, 3, 8*9,32*9,0,0,5,50,1,StereoSGBM::MODE_SGBM);

	left_matcher->setSpeckleWindowSize(200);
	left_matcher->setSpeckleRange(64);
//	left_matcher->setTextureThreshold(10);
//	Ptr<ximgproc::DisparityWLSFilter>  wls_filter = ximgproc::createDisparityWLSFilter(left_matcher);
//	Ptr<StereoMatcher> right_matcher = ximgproc::createRightMatcher(left_matcher);
//
//	Mat right_disp,filtered_disp;

	left_matcher->compute(rect1, rect2, left_disp);
//	right_matcher->compute(rect2, rect1, right_disp);
//
//	wls_filter->setLambda(8000);
//	wls_filter->setSigmaColor(3.0);
//	wls_filter->filter(left_disp, rect1, filtered_disp, right_disp);

//	displayDispMap(filtered_disp);

	return left_disp;
}

long calculateDescriptorCost(int x1, int y, int x2, int window_size, std::vector<KeyPoint>& rect1_keypoints, std::vector<KeyPoint>& rect2_keypoints, Mat& rect1_descriptors, Mat& rect2_descriptors)
{
	//A window around the given point
	long cost = 0;
	for (int i = -window_size; i <= window_size; i++)
	{
		int a = 0;
		for (int j = -window_size; j <= window_size; j++)
		{
			//What THE IF
			if (((x1 + i) >= rect1_keypoints[0].pt.x) &&
				((x1 + i) <= rect1_keypoints[rect1_keypoints.size() - 1].pt.x) &&
				((y + j) >= rect1_keypoints[0].pt.y) &&
				((y + j) <= rect1_keypoints[rect1_keypoints.size() - 1].pt.y)) //If a left key point
			{
				if (((x2 + i) >= rect2_keypoints[0].pt.x &&
					(x2 + i) <= rect2_keypoints[rect2_keypoints.size() - 1].pt.x &&
					(y + j) >= rect2_keypoints[0].pt.y &&
					(y + j) <= rect2_keypoints[rect2_keypoints.size() - 1].pt.y)) //If a right key point
				{
					int desc_index1 = ((x1 + i) - rect1_keypoints[0].pt.x) * (rect1_keypoints[rect1_keypoints.size() - 1].pt.y - rect1_keypoints[0].pt.y + 1) + ((y + j) - rect1_keypoints[0].pt.y);//Descriptor index of the point in the first image
					int desc_index2 = ((x2 + i) - rect2_keypoints[0].pt.x) * (rect2_keypoints[rect2_keypoints.size() - 1].pt.y - rect2_keypoints[0].pt.y + 1) + ((y + j) - rect2_keypoints[0].pt.y);//Descriptor index of the point in the second image
					cost = cost + norm(rect1_descriptors.row(desc_index1), rect2_descriptors.row(desc_index2), 2);
				}
			}
		}
	}
	cost = cost / ((2 * window_size + 1) * (2 * window_size + 1)); //average cost in the window

	return cost;
}

int calculateCorrespondingPoint(int x, int y, int max_disp, int window_size, std::vector<KeyPoint>& rect1_keypoints, std::vector<KeyPoint>& rect2_keypoints, Mat& rect1_descriptors, Mat& rect2_descriptors) {
	long minimum_cost = 1e9;
	int min_cost_index = 0;
	long cur_cost = minimum_cost;
	for (int i = x - max_disp; i <= x; i++)
	{
		cur_cost = calculateDescriptorCost(x, y, i, window_size, rect1_keypoints, rect2_keypoints, rect1_descriptors, rect2_descriptors);
		if (cur_cost < minimum_cost)
		{
			minimum_cost = cur_cost;
			min_cost_index = i;
		}
	}
	if (minimum_cost == 0) //if corresponding point is the same as the other one
		return x;
	else
		return min_cost_index;
}

Mat computeDisparityMapORB(Mat rect1, Mat rect2, int max_disp, int window_size)
{
	std::vector<KeyPoint> rect1_keypoints;
	std::vector<KeyPoint> rect2_keypoints;

	for (int i = 0; i < rect1.size().width; i++)
	{
		for (int j = 0; j < rect1.size().height; j++)
		{
			rect1_keypoints.push_back(KeyPoint(i, j, 1));
			rect2_keypoints.push_back(KeyPoint(i, j, 1));
		}
	}
	Ptr<DescriptorExtractor> descriptorExtractor = ORB::create();
	Mat rect1_descriptors, rect2_descriptors;
	descriptorExtractor->compute(rect1, rect1_keypoints, rect1_descriptors); //Default best 500 keypoints
	descriptorExtractor->compute(rect2, rect2_keypoints, rect2_descriptors);
	Mat disparity_map = Mat(rect1.size().height, rect1.size().width, CV_8UC1, Scalar(0));


	for (int i = max_disp + 1; i < rect1.size().width; i++)
	{
		std::cout << "i =   " << i << ", out of " << rect1.size().width << std::endl;
		for (int j = 0; j < rect1.size().height; j++)
		{

			if (i >= rect1_keypoints[0].pt.x && i <= rect1_keypoints[rect1_keypoints.size() - 1].pt.x && j >= rect1_keypoints[0].pt.y && j <= rect1_keypoints[rect1_keypoints.size() - 1].pt.y)
			{
				int right_corresponding_point = calculateCorrespondingPoint(i, j, max_disp, window_size, rect1_keypoints, rect2_keypoints, rect1_descriptors, rect2_descriptors);

				int disparity_value = abs(i - right_corresponding_point);
				disparity_map.at<uchar>(j, i) = disparity_value; //if doesnt work change to i,j
			}
		}
	}
	Mat blurred_disp_map;
	medianBlur(disparity_map, blurred_disp_map, 31);
	blurred_disp_map.convertTo(blurred_disp_map, CV_32F, 128.0);
	displayDispMap(blurred_disp_map);
	return blurred_disp_map;
}