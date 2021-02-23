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

class AbstractStereoMatcher
{
public:
	AbstractStereoMatcher();

	virtual ~AbstractStereoMatcher() {};

	// computes disparity map
	virtual Mat computeDisparityMap(Mat rect1_, Mat rect2_) = 0;

	static AbstractStereoMatcher* Create(String stereoMatcher);

protected:
	const Mat rect1_;
	const Mat rect2_;
};

class BMMatcher : public AbstractStereoMatcher
{
	public:
	virtual Mat computeDisparityMap(Mat rect1_, Mat rect2_);
};

class SGBMMatcher : public AbstractStereoMatcher
{
public:
	virtual Mat computeDisparityMap(Mat rect1_, Mat rect2_);
};

class ORBMatcher : public AbstractStereoMatcher
{
public:
	virtual Mat computeDisparityMap(Mat rect1_, Mat rect2_);
private:
	long calculateDescriptorCost(int x1, int y, int x2, int window_size, std::vector<KeyPoint>& rect1_keypoints, std::vector<KeyPoint>& rect2_keypoints, Mat& rect1_descriptors, Mat& rect2_descriptors);
	int calculateCorrespondingPoint(int x, int y, int max_disp, int window_size, std::vector<KeyPoint>& rect1_keypoints, std::vector<KeyPoint>& rect2_keypoints, Mat& rect1_descriptors, Mat& rect2_descriptors);
};

void displayDispMap(Mat disp_map);