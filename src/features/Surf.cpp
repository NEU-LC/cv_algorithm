#include "Surf.h"

namespace MCV{
    void extractSurf(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints)
    {
        assert(!src.empty());
        cv::Mat img  = src.getMat();
        if(img.channels() == 3)
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::xfeatures2d::SURF> surfPtr = cv::xfeatures2d::SURF::create(800);
        surfPtr -> detect(img, keypoints);
    }

    void extractSurfAndDescription(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints, cv::Mat &des)
    {
        assert(!src.empty());
        cv::Mat img  = src.getMat();
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::xfeatures2d::SURF> surfPtr = cv::xfeatures2d::SURF::create(800);
        surfPtr -> detectAndCompute(img, cv::Mat(), keypoints, des);
    }

    void extractCudaSurf(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints)
    {
        assert(!src.empty());
        cv::cuda::GpuMat gpuImg = src.getGpuMat();

        if(gpuImg.channels() == 3)
            cv::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2GRAY);

        cv::cuda::SURF_CUDA surfgpu(100, 4, 3);
        surfgpu(gpuImg, cv::cuda::GpuMat(), keypoints);
    }

    void extractCudaSurfAndDescription(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &des)
    {
        assert(!src.empty());
        cv::cuda::GpuMat gpuImg = src.getGpuMat();

        if(gpuImg.channels() == 3)
            cv::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2GRAY);

        cv::cuda::SURF_CUDA surfgpu(100, 4, 3);
        surfgpu(gpuImg, cv::cuda::GpuMat(), keypoints, des);
         
    }

}