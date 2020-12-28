#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "features/Surf.h"
int main(int argc, char** argv)
{
    std::string path = argv[1];
    cv::cuda::GpuMat gMat;
    gMat.upload(cv::imread(path, cv::IMREAD_GRAYSCALE));
    if(gMat.empty())
    {
        std::cerr << "image is null!" << std::endl;
        return EXIT_FAILURE;
    }
    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
    std::vector<cv::KeyPoint> gpuSurfKeypoints;
    cv::cuda::GpuMat gpuSurfDes;
    MCV::extractCudaSurf(gMat, gpuSurfKeypoints);
    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
    double surfTime = (t2 - t1).total_microseconds() * 1e-6;
    std::cout << "extract gpu surf features cost: " << surfTime << "s" << std::endl;

    boost::posix_time::ptime t3 = boost::posix_time::microsec_clock::local_time();
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> surfKeypoints;
    cv::Mat des;
    MCV::extractSurf(img, surfKeypoints);
    boost::posix_time::ptime t4 = boost::posix_time::microsec_clock::local_time();
    double gpusuftTime = (t4 - t3).total_microseconds() * 1e-6;
    std::cout << "extract surf features cost: " << gpusuftTime << "s" << std::endl;

    cv::Mat showSurf, showGpuSurf;
    cv::Mat gMatDown;
    // cuda的内存不能是每个字节操作，需要进行一下内存转换，读取也是同样的道理
    gMat.download(gMatDown);
    cv::drawKeypoints(gMatDown, gpuSurfKeypoints, showGpuSurf);
    cv::drawKeypoints(img, surfKeypoints, showSurf);

    cv::imshow("show surf", showSurf);
    cv::imshow("show gpu surf", showGpuSurf);

    cv::waitKey();
    
}