#include "Camera.h"

Camera::Camera(float fx, float fy, float cx, float cy,
               float k1, float k2, float p1, float p2, float k3,
               float baseline)
    : fx(fx), fy(fy), cx(cx), cy(cy),
      k1(k1), k2(k2), p1(p1), p2(p2), k3(k3),
      mBaseline(baseline)
{
    // 初始化内参矩阵 K (Eigen 方式)
    K << fx,  0, cx,
          0, fy, cy,
          0,  0,  1;

    // 提前计算内参逆矩阵
    K_inv = K.inverse();
}

Eigen::Vector2f Camera::Camera2Pixel(const Eigen::Vector3f &P_c)
{
    return Eigen::Vector2f(
        fx * P_c.x() / P_c.z() + cx,
        fy * P_c.y() / P_c.z() + cy
    );
}

Eigen::Vector3f Camera::Pixel2Camera(const Eigen::Vector2f &p_p)
{
    return Eigen::Vector3f(
        (p_p.x() - cx) / fx,
        (p_p.y() - cy) / fy,
        1.0f
    );
}

Eigen::Vector3f Camera::World2Camera(const Eigen::Vector3f &P_w, const Eigen::Matrix4f &T_cw)
{
    // 使用齐次坐标矩阵乘法直接完成变换: T_cw * P_w_homogeneous
    Eigen::Vector4f P_w_homo(P_w.x(), P_w.y(), P_w.z(), 1.0f);
    Eigen::Vector4f P_c_homo = T_cw * P_w_homo;
    return P_c_homo.head<3>();
}

Eigen::Vector3f Camera::Camera2World(const Eigen::Vector3f &P_c, const Eigen::Matrix4f &T_wc)
{
    Eigen::Vector4f P_c_homo(P_c.x(), P_c.y(), P_c.z(), 1.0f);
    Eigen::Vector4f P_w_homo = T_wc * P_c_homo;
    return P_w_homo.head<3>();
}

Eigen::Vector2f Camera::UndistortPoint(const Eigen::Vector2f &p_p)
{
    cv::Mat cv_K = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat cv_D = (cv::Mat_<float>(5, 1) << k1, k2, p1, p2, k3);

    cv::Mat mat_p(1, 1, CV_32FC2);
    mat_p.at<cv::Vec2f>(0, 0) = cv::Vec2f(p_p.x(), p_p.y());
    
    cv::Mat undistorted;
    cv::undistortPoints(mat_p, undistorted, cv_K, cv_D, cv::Mat(), cv_K);

    cv::Vec2f res = undistorted.at<cv::Vec2f>(0, 0);
    return Eigen::Vector2f(res[0], res[1]);
}