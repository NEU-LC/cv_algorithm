#ifndef CV_CORE_SURF_H
#define CV_CORE_SURF_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace MCV{

    void extractSurf(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints);
    void extractSurfAndDescription(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints, cv::Mat &des);
    void extractCudaSurf(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints);
    void extractCudaSurfAndDescription(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints, cv::Mat &des);
}

#endif