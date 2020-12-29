//
// Created by chao on 2020/12/29.
//

#ifndef CV_CORE_SIFT_H
#define CV_CORE_SIFT_H

#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <SiftGPU.h>

namespace MCV{

    void initSiftGpu();
    void extractSift(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints, int nPoints);
    void extractSiftGpu(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints, cv::Mat &description);
}
#endif //CV_CORE_SIFT_H
