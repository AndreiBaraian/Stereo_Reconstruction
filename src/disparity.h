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

Mat computeDisparityMapBM(Mat rect1, Mat rect2)
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

Mat computeDisparityMapSGBM(Mat rect1, Mat rect2)
{
    Mat left_disp;

    int ndisparities = 15;
    int SADWindowSize = 250;

    Ptr<StereoSGBM> left_matcher = StereoSGBM::create(ndisparities, SADWindowSize, 3, 8*9,32*9,0,0,5,50,1,StereoSGBM::MODE_SGBM);
    //Ptr<ximgproc::DisparityWLSFilter>  wls_filter = ximgproc::createDisparityWLSFilter(left_matcher);
    //Ptr<StereoMatcher> right_matcher = ximgproc::createRightMatcher(left_matcher);

    //Mat right_disp,filtered_disp;

    left_matcher->compute(rect1, rect2, left_disp);
    //right_matcher->compute(rect2, rect1, right_disp);

    //wls_filter->setLambda(8000);
    //wls_filter->setSigmaColor(3.0);
    //wls_filter->filter(left_disp, rect1, filtered_disp, right_disp);

    displayDispMap(left_disp);

    return left_disp;
}


Mat filterDispMap(Mat rect1, Mat rect2)
{
    Mat disp_map_l ;//= Mat(rect1.rows, rect2.cols, CV_16S);
    Mat disp_map_r ;//= Mat(rect1.rows, rect2.cols, CV_16S);

    int ndisparities = 256;
    int SADWindowSize = 15;

    Ptr<StereoSGBM> left_sbm = StereoSGBM::create(ndisparities, SADWindowSize, 3, 8*9,32*9,0,0,5,50,1,StereoSGBM::MODE_SGBM);

    //for filtering
    Mat filter_disp_map = Mat(rect1.rows, rect2.cols, CV_16S);
    auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_sbm);
    auto right_sbm = cv::ximgproc::createRightMatcher(left_sbm);

    left_sbm-> compute(rect1, rect2,disp_map_l);
    right_sbm->compute(rect2,rect1, disp_map_r);

//    displayDispMap(disp_map_r);
    //right disp map is all black needs a fix
    wls_filter->setLambda(10000);
    wls_filter->setSigmaColor(2.0);

    wls_filter->filter(disp_map_l,rect1,filter_disp_map, disp_map_r) ;

    displayDispMap(filter_disp_map);

    return filter_disp_map;
}

