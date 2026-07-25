#include "Camera.h"

Camera::Camera(float fx, float fy, float cx, float cy,
               float k1, float k2, float p1, float p2, float k3,
               float baseline)
    : fx(fx), fy(fy), cx(cx), cy(cy),
      k1(k1), k2(k2), p1(p1), p2(p2), k3(k3),
      mBaseline(baseline)
{

    // 初始化内参矩阵 K
    K = (cv::Mat_<float>(3, 3) << fx, 0, cx,
         0, fy, cy,
         0, 0, 1);
    // 提前计算内参逆矩阵
    K_inv = K.inv();
}

cv::Point2f Camera::Camera2Pixel(const cv::Mat &P_c)
{
    return cv::Point2f(
        fx * P_c.at<float>(0, 0) / P_c.at<float>(2, 0) + cx,
        fy * P_c.at<float>(1, 0) / P_c.at<float>(2, 0) + cy);
}

cv::Point2f Camera::Camera2Pixel(const cv::Point3f &P_c)
{
    return cv::Point2f(
        fx * P_c.x / P_c.z + cx,
        fy * P_c.y / P_c.z + cy);
}

cv::Point3f Camera::Pixel2Camera(const cv::Point2f &p_p)
{
    return cv::Point3f(
        (p_p.x - cx) / fx,
        (p_p.y - cy) / fy,
        1.0f);
}

cv::Point3f Camera::World2Camera(const cv::Point3f &P_w, const cv::Mat &T_cw)
{
    cv::Mat P_w_mat = (cv::Mat_<float>(4, 1) << P_w.x, P_w.y, P_w.z, 1.0f);
    cv::Mat P_c_mat = T_cw * P_w_mat;
    return cv::Point3f(P_c_mat.at<float>(0, 0), P_c_mat.at<float>(1, 0), P_c_mat.at<float>(2, 0));
}

cv::Point3f Camera::Camera2World(const cv::Point3f &P_c, const cv::Mat &T_wc)
{
    cv::Mat P_c_mat = (cv::Mat_<float>(4, 1) << P_c.x, P_c.y, P_c.z, 1.0f);
    cv::Mat P_w_mat = T_wc * P_c_mat;
    return cv::Point3f(P_w_mat.at<float>(0, 0), P_w_mat.at<float>(1, 0), P_w_mat.at<float>(2, 0));
}

cv::Point2f Camera::UndistortPoint(const cv::Point2f &p_p)
{
    cv::Mat mat_p(1, 1, CV_32FC2);
    mat_p.at<cv::Vec2f>(0, 0) = cv::Vec2f(p_p.x, p_p.y);
    cv::Mat undistorted;

    cv::Mat D = (cv::Mat_<float>(5, 1) << k1, k2, p1, p2, k3);
    cv::undistortPoints(mat_p, undistorted, K, D, cv::Mat(), K);

    return cv::Point2f(undistorted.at<cv::Vec2f>(0, 0)[0], undistorted.at<cv::Vec2f>(0, 0)[1]);
}