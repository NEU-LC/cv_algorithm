#include "mcvPnP.h"

namespace MCV{
     bool findEssentialFromEightPoints(std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, cv::Mat &E)
     {
        assert(pts1.size() >= 0 && pts2.size() >= 0 && pts1.size() == pts2.size());

        int num = pts1.size();
        double t, scale1, scale2;
        cv::Point2f m1c(0,0), m2c(0,0);

        std::vector<cv::Point2f> n_pts1, n_pts2;
        n_pts1.resize(num);
        n_pts2.resize(num);

        for(int i = 0; i < num; ++i)
        {
            m1c += pts1[i];
            m2c += pts2[i];
        }
        

        for(int i = 0; i < num; ++i)
        {
            n_pts1[i] = cv::Point2f(pts1[i].x/m1c.x, pts1[i].y/m1c.y);
            n_pts2[i] = cv::Point2f(pts2[i].x/m2c.x, pts2[i].y/m2c.y);
        }

        cv::Mat A(num, 9, CV_32F);
        cv::Vec3f W;
        cv::Matx33f U;
        cv::Matx33f Vt;
        
        for(int i = 0; i < num; ++i)
        {
            const float u1 = n_pts1[i].x;
            const float v1 = n_pts1[i].y;
            const float u2 = n_pts2[i].x;
            const float v2 = n_pts2[i].y;
            A.at<float>(i,0) = u2*u1;
            A.at<float>(i,1) = u2*v1;
            A.at<float>(i,2) = u2;
            A.at<float>(i,3) = v2*u1;
            A.at<float>(i,4) = v2*v1;
            A.at<float>(i,5) = v2;
            A.at<float>(i,6) = u1;
            A.at<float>(i,7) = v1;
            A.at<float>(i,8) = 1;
        }
        cv::SVD::compute(A, W, U, Vt);
        W[2] = 0.;
        cv::Matx33f F0 = U*cv::Matx33f::diag(W) * Vt;
        cv::Mat(F0).copyTo(E); 


     }
}