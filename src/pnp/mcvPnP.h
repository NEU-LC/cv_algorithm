#ifndef MY_CV_PNP_H
#define MY_CV_PNP_H

#include <iostream>
#include <opencv2/core/core.hpp>

namespace MCV{
    bool findEssentialFromEightPoints(std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, cv::Mat &E);    
}

#endif