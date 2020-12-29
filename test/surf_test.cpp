#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "features/Surf.h"
#include "features/Sift.h"
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
    // surf bu cuda opencv
    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
    std::vector<cv::KeyPoint> gpuSurfKeypoints;
    cv::cuda::GpuMat gpuSurfDes;
    MCV::extractCudaSurf(gMat, gpuSurfKeypoints);
    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
    double surfTime = (t2 - t1).total_microseconds() * 1e-6;
    std::cout << "extract gpu surf features cost: " << surfTime << "s" << std::endl;

    boost::posix_time::ptime t1_ = boost::posix_time::microsec_clock::local_time();
    gpuSurfKeypoints.clear();
    MCV::extractCudaSurf(gMat, gpuSurfKeypoints);
    boost::posix_time::ptime t2_ = boost::posix_time::microsec_clock::local_time();
    double surfTime_ = (t2_ - t1_).total_microseconds() * 1e-6;
    std::cout << "The second extract gpu surf features cost: " << surfTime_ << "s" << std::endl;

    // surf by opencv
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

    // siftGpu
    boost::posix_time::ptime t5 = boost::posix_time::microsec_clock::local_time();
    std::vector<cv::KeyPoint> siftGpuKeypoints;
    cv::Mat siftDes;
    MCV::extractSiftGpu(img, siftGpuKeypoints, siftDes);
    std::cout << "sift features num: " << siftGpuKeypoints.size() << std::endl;
    boost::posix_time::ptime t6 = boost::posix_time::microsec_clock::local_time();
    double siftGpuTime = (t6 - t5).total_microseconds() * 1e-6;
    std::cout << "extract gpu sift features cost: " << siftGpuTime << "s" << std::endl;

    // why the second time cost of extracting sift features by siftGpu is less than the first???
    boost::posix_time::ptime t7 = boost::posix_time::microsec_clock::local_time();
    MCV::extractSiftGpu(img, siftGpuKeypoints, siftDes);
    std::cout << "sift features num: " << siftGpuKeypoints.size() << std::endl;
    boost::posix_time::ptime t8 = boost::posix_time::microsec_clock::local_time();
    double siftGpuTime2 = (t8 - t7).total_microseconds() * 1e-6;
    std::cout << "extract gpu sift features 2 cost: " << siftGpuTime2 << "s" << std::endl;

    cv::Mat showGpuSift;
    cv::drawKeypoints(img, siftGpuKeypoints, showGpuSift);
    cv::imshow("show gpu sift", showGpuSift);

    cv::waitKey();
    
}