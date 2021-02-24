#include "disparity.h"

AbstractStereoMatcher::AbstractStereoMatcher(){}

AbstractStereoMatcher* AbstractStereoMatcher::Create(String matcher_type)
{
	if (matcher_type == "BM")
		return new BMMatcher();
	else if (matcher_type == "SGBM")
		return new SGBMMatcher();
	else if (matcher_type == "ORB")
		return new ORBMatcher();
}

Mat BMMatcher::computeDisparityMap(Mat rect1_, Mat rect2_)
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

	sbm->compute(rect1_, rect2_, disp_map);

	// displayDispMap(disp_map);

	return disp_map;
}

Mat SGBMMatcher::computeDisparityMap(Mat rect1_, Mat rect2_)
{
	Mat left_disp;

	int ndisparities = 15;
	int SADWindowSize = 250;

	Ptr<StereoSGBM> left_matcher = StereoSGBM::create(ndisparities, SADWindowSize, 3, 8 * 9, 32 * 9, 0, 0, 5, 50, 1, StereoSGBM::MODE_SGBM);

	left_matcher->setSpeckleWindowSize(200);
	left_matcher->setSpeckleRange(64);
	//	left_matcher->setTextureThreshold(10);
	//	Ptr<ximgproc::DisparityWLSFilter>  wls_filter = ximgproc::createDisparityWLSFilter(left_matcher);
	//	Ptr<StereoMatcher> right_matcher = ximgproc::createRightMatcher(left_matcher);
	//
	//	Mat right_disp,filtered_disp;

	left_matcher->compute(rect1_, rect2_, left_disp);
	//	right_matcher->compute(rect2, rect1, right_disp);
	//
	//	wls_filter->setLambda(8000);
	//	wls_filter->setSigmaColor(3.0);
	//	wls_filter->filter(left_disp, rect1, filtered_disp, right_disp);

	//	displayDispMap(filtered_disp);

	return left_disp;
}

Mat ORBMatcher::computeDisparityMap(Mat rect1_, Mat rect2_)
{
	int max_disp = 400;
	int window_size = 0;
	std::vector<KeyPoint> rect1_keypoints;
	std::vector<KeyPoint> rect2_keypoints;

	for (int i = 0; i < rect1_.size().width; i++)
	{
		for (int j = 0; j < rect1_.size().height; j++)
		{
			rect1_keypoints.push_back(KeyPoint(i, j, 1));
			rect2_keypoints.push_back(KeyPoint(i, j, 1));
		}
	}
	Ptr<DescriptorExtractor> descriptorExtractor = ORB::create();
	Mat rect1_descriptors, rect2_descriptors;
	descriptorExtractor->compute(rect1_, rect1_keypoints, rect1_descriptors); //Default best 500 keypoints
	descriptorExtractor->compute(rect2_, rect2_keypoints, rect2_descriptors);
	Mat disparity_map = Mat(rect1_.size().height, rect1_.size().width, CV_8UC1, Scalar(0));


	for (int i = max_disp + 1; i < rect1_.size().width; i++)
	{
		// std::cout << "i =   " << i << ", out of " << rect1_.size().width << std::endl;
		for (int j = 0; j < rect1_.size().height; j++)
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

long ORBMatcher::calculateDescriptorCost(int x1, int y, int x2, int window_size, std::vector<KeyPoint>& rect1_keypoints, std::vector<KeyPoint>& rect2_keypoints, Mat& rect1_descriptors, Mat& rect2_descriptors)
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

int ORBMatcher::calculateCorrespondingPoint(int x, int y, int max_disp, int window_size, std::vector<KeyPoint>& rect1_keypoints, std::vector<KeyPoint>& rect2_keypoints, Mat& rect1_descriptors, Mat& rect2_descriptors) {
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

	normalize(disp_map, result, 0, 256, NORM_MINMAX, CV_8U);

	imshow("disparity map", result);
	waitKey(0);
}