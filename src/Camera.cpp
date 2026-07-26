#include "Camera.h"

Camera::Camera(float fx, float fy, float cx, float cy,
               float k1, float k2, float p1, float p2, float k3,
               float baseline)
    : fx(fx), fy(fy), cx(cx), cy(cy),
      k1(k1), k2(k2), p1(p1), p2(p2), k3(k3),
      mBaseline(baseline)
{
    // 初始化内参矩阵 K (Eigen 方式)
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;

    // 提前计算内参逆矩阵
    K_inv = K.inverse();
}

Eigen::Vector2f Camera::Camera2Pixel(const Eigen::Vector3f &P_c)
{
    return Eigen::Vector2f(
        fx * P_c.x() / P_c.z() + cx,
        fy * P_c.y() / P_c.z() + cy);
}

Eigen::Vector3f Camera::Pixel2Camera(const Eigen::Vector2f &p_p)
{
    return Eigen::Vector3f(
        (p_p.x() - cx) / fx,
        (p_p.y() - cy) / fy,
        1.0f);
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

bool Camera::TriangulateDLT(const Eigen::Matrix4f &T_cw1, const Eigen::Matrix4f &T_cw2,
                            const Eigen::Vector2f &pt1_norm, const Eigen::Vector2f &pt2_norm,
                            Eigen::Vector3f &P_w)
{
    // 归一化坐标下，投影矩阵直接取 T_cw 的前 3x4 块 [R | t]
    Eigen::Matrix<float, 3, 4> P1 = T_cw1.block<3, 4>(0, 0);
    Eigen::Matrix<float, 3, 4> P2 = T_cw2.block<3, 4>(0, 0);

    Eigen::Vector4f p1_1 = P1.row(0), p1_2 = P1.row(1), p1_3 = P1.row(2);
    Eigen::Vector4f p2_1 = P2.row(0), p2_2 = P2.row(1), p2_3 = P2.row(2);

    // 构建方程组 A * X = 0
    Eigen::Matrix4f A;
    A.row(0) = pt1_norm.x() * p1_3 - p1_1;
    A.row(1) = pt1_norm.y() * p1_3 - p1_2;
    A.row(2) = pt2_norm.x() * p2_3 - p2_1;
    A.row(3) = pt2_norm.y() * p2_3 - p2_2;

    // SVD 求解
    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f X_homo = svd.matrixV().col(3);

    if (std::abs(X_homo.w()) < 1e-6f)
        return false;

    // 齐次坐标归一化
    P_w = X_homo.head<3>() / X_homo.w();

    // 正深度检查 (Check Positive Depth)
    Eigen::Vector3f P_c1 = T_cw1.block<3, 3>(0, 0) * P_w + T_cw1.block<3, 1>(0, 3);
    Eigen::Vector3f P_c2 = T_cw2.block<3, 3>(0, 0) * P_w + T_cw2.block<3, 1>(0, 3);

    return (P_c1.z() > 0.0f && P_c2.z() > 0.0f);
}