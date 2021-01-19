#include<iostream>
#include<fstream>
#include<opencv2/opencv.hpp>

void readCameraCalib(std::string filename, cv::Mat& cameraMatrix0, cv::Mat& cameraMatrix1, double& doffs, double& baseline, int& width, int& height, int& ndisp, int& isint, int& vmin, int& vmax, double& dyavg, double& dymax)
{
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        int i = 0;
        while (std::getline(file, line)) {
            std::string curStr = line.c_str();
            curStr = curStr.substr(curStr.find_first_of('=') + 1);

            if (i == 0)
            {
                curStr = curStr.substr(curStr.find_first_of('[') + 1);
                curStr.pop_back();
                curStr.erase(std::remove(curStr.begin(), curStr.end(), ';'), curStr.end());
                std::istringstream is(curStr);
                std::string part;
                int j = 0;
                double x[3][3];
                while (getline(is, part, ' '))
                {
                    x[j / 3][j % 3] = std::stod(part);
                    j++;
                }
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        cameraMatrix0.at<double>(j, k) = x[j][k];
                    }
                }
            }
            else if (i == 1)
            {
                curStr = curStr.substr(curStr.find_first_of('[') + 1);
                curStr.pop_back();
                curStr.erase(std::remove(curStr.begin(), curStr.end(), ';'), curStr.end());
                std::istringstream is(curStr);
                std::string part;
                int j = 0;
                double x[3][3];
                while (getline(is, part, ' '))
                {
                    x[j / 3][j % 3] = std::stod(part);
                    j++;
                }
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        cameraMatrix1.at<double>(j, k) = x[j][k];
                    }
                }
            }
            else if (i == 2)
            {
                doffs = std::stod(curStr);
            }
            else if (i == 3)
            {
                baseline = std::stod(curStr);
            }
            else if (i == 4)
            {
                width = std::stoi(curStr);
            }
            else if (i == 5)
            {
                height = std::stoi(curStr);
            }
            else if (i == 6)
            {
                ndisp = std::stoi(curStr);
            }
            else if (i == 7)
            {
                isint = std::stoi(curStr);
            }
            else if (i == 8)
            {
                vmin = std::stoi(curStr);
            }
            else if (i == 9)
            {
                vmax = std::stoi(curStr);
            }
            else if (i == 10)
            {
                dyavg = std::stod(curStr);
            }
            else if (i == 11)
            {
                dymax = std::stod(curStr);
            }
            i++;
        }
        file.close();
    }
}

void rectifyImages(cv::Mat img0, cv::Mat img1, cv::Mat& recImg0, cv::Mat& recImg1, cv::Mat cameraMatrix0, cv::Mat cameraMatrix1, double& baseline, int& width, int& height)
{
    cv::Mat T = cv::Mat::zeros(3, 1, CV_64F);
    T.at<double>(1, 0) = baseline * cameraMatrix0.at<double>(0, 0);

    cv::Mat R1 = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat R2 = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);

    cv::Rect temp[2];

    stereoRectify(cameraMatrix0, cv::Mat::zeros(1, 4, CV_64F), cameraMatrix1, cv::Mat::zeros(1, 4, CV_64F), cv::Size(width, height), cv::Mat::eye(3, 3, CV_64F), T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, cv::Size(), &temp[0], &temp[1]);

    cv::Mat recMap[2][2];

    initUndistortRectifyMap(cameraMatrix0, cv::Mat::zeros(1, 4, CV_64F), R1, P1, cv::Size(width, height), CV_16SC2, recMap[0][0], recMap[0][1]);
    initUndistortRectifyMap(cameraMatrix1, cv::Mat::zeros(1, 4, CV_64F), R2, P2, cv::Size(width, height), CV_16SC2, recMap[1][0], recMap[1][1]);

    remap(img0, recImg0, recMap[0][0], recMap[0][1], cv::INTER_LINEAR);
    remap(img1, recImg1, recMap[1][0], recMap[1][1], cv::INTER_LINEAR);
}

void visualizeRectified(cv::Mat recImg0, cv::Mat recImg1, int width, int height)
{
    /*
    cv::Mat canvas;
    double sf;
    int w, h;

    sf = 300. / MAX(width, height);
    w = cvRound(width * sf);
    h = cvRound(height * sf);
    canvas.create(h, 2 * w, CV_8UC3);
    
    cv::Mat canvasPart0 = canvas(cv::Rect(0, 0, w, h));
    cv::Mat canvasPart1 = canvas(cv::Rect(w, 0, w, h));
    resize(recImg0, canvasPart0, canvasPart0.size(), 0, 0, cv::INTER_AREA);
    resize(recImg1, canvasPart1, canvasPart1.size(), 0, 0, cv::INTER_AREA);
    */
    cv::Mat concat_img;
    hconcat(recImg0, recImg1, concat_img);
    for (int i = 0; i < concat_img.rows; i += 64)//Change 64 to another value to change the total number of lines
    {
        line(concat_img, cv::Point(0, i), cv::Point(concat_img.cols, i), cv::Scalar(0, 255, 0), 8, cv::LINE_4);//Change 8 to another value to change the thickness of the lines
    }
    cv::namedWindow("Display frame", cv::WINDOW_FREERATIO);
    imshow("Display frame", concat_img);
    cv::waitKey(0);
}