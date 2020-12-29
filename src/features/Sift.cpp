//
// Created by chao on 2020/12/29.
//

#include "Sift.h"
#include <GL/gl.h>
SiftGPU sift_gpu;
namespace MCV{

    void initSiftGpu(){
        char * sift_gpu_argv[] ={(char*)"-t", (char*)"0", (char*)"-v", (char*)"0", (char*)"-cuda"};
        sift_gpu.ParseParam(4, sift_gpu_argv);
        int support = sift_gpu.CreateContextGL();
        if(support != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
            return;
        } else {
            printf("SiftGPU supported\n");
        }

    }
    void extractSift(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints, int nPoints){
        cv::Mat m_src = src.getMat();
        cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> ptr = cv::xfeatures2d::SIFT::create(nPoints);
        ptr -> detect(m_src, keypoints);
    }

    void extractSiftGpu(cv::InputArray &src, std::vector<cv::KeyPoint> &keypoints, cv::Mat &description){
        initSiftGpu();
        cv::Mat m_src = src.getMat();
        sift_gpu.RunSIFT(m_src.cols, m_src.rows, m_src.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
        size_t num = sift_gpu.GetFeatureNum();
        std::vector<SiftGPU::SiftKeypoint> gpuKeypoints(num);
        description = cv::Mat::zeros(num, 128, CV_32F);

        std::vector<float > vDes;
        vDes.reserve(128*num);

        auto des = new float[128*num];
        sift_gpu.GetFeatureVector(&gpuKeypoints[0], &des[0]);

        keypoints.resize(num);
        for(int i = 0; i < num; ++i)
        {
            if(gpuKeypoints[i].x >= m_src.cols || gpuKeypoints[i].x <= 0 ||
                gpuKeypoints[i].y >= m_src.rows || gpuKeypoints[i].y <= 0)
                continue;
            cv::KeyPoint kp;
            kp.pt = cv::Point2f(gpuKeypoints[i].x, gpuKeypoints[i].y);
            keypoints[i] = kp;
            vDes.insert(vDes.end(), des+i*128, des+(i+1)*128);

        }
        int fn = vDes.size() / 128;
        //cv::KeyPointsFilter::runByImageBorder(keypoints, m_src.size(), 3)
        cv::Mat matdes = cv::Mat(cv::Size(128, fn), CV_32FC1, vDes.data());
        description = matdes.clone();
    }
}
