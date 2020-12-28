//
// Created by chao on 2020/12/20.
//

#ifndef CV_CORE_SINGLE_TRIANGULATION_H
#define CV_CORE_SINGLE_TRIANGULATION_H

#include <opencv2/opencv.hpp>
namespace MCV{
    cv::Matx33d single_triangleulation(std::vector<cv::Point2f> &pts2d, std::vector<cv::Matx44d> &camPose);
}

#endif //CV_CORE_SINGLE_TRIANGULATION_H

