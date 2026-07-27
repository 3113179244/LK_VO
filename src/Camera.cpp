#include "Camera.h"

Camera::Camera()
    : fx(Config::g_dFx0),
      fy(Config::g_dFy0),
      cx(Config::g_dCx0),
      cy(Config::g_dCy0),
      k1(Config::g_dK1_0),
      k2(Config::g_dK2_0),
      p1(Config::g_dP1_0),
      p2(Config::g_dP2_0),
      k3(0.0),
      mBaseline((Config::g_mBodyTCam0.inverse() * Config::g_mBodyTCam1).block<3, 1>(0, 3).norm())
{
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;

    K_inv = K.inverse();
}

Eigen::Vector2d Camera::Camera2Pixel(const Eigen::Vector3d &P_c)
{
    return Eigen::Vector2d(
        fx * P_c.x() / P_c.z() + cx,
        fy * P_c.y() / P_c.z() + cy);
}

Eigen::Vector3d Camera::Pixel2Camera(const Eigen::Vector2d &p_p)
{
    return Eigen::Vector3d(
        (p_p.x() - cx) / fx,
        (p_p.y() - cy) / fy,
        1.0);
}

Eigen::Vector3d Camera::World2Camera(const Eigen::Vector3d &P_w, const Eigen::Matrix4d &T_cw)
{
    Eigen::Vector4d P_w_homo(P_w.x(), P_w.y(), P_w.z(), 1.0);
    Eigen::Vector4d P_c_homo = T_cw * P_w_homo;
    return P_c_homo.head<3>();
}

Eigen::Vector3d Camera::Camera2World(const Eigen::Vector3d &P_c, const Eigen::Matrix4d &T_wc)
{
    Eigen::Vector4d P_c_homo(P_c.x(), P_c.y(), P_c.z(), 1.0);
    Eigen::Vector4d P_w_homo = T_wc * P_c_homo;
    return P_w_homo.head<3>();
}

Eigen::Vector2d Camera::UndistortPoint(const Eigen::Vector2d &p_p)
{
    cv::Mat cv_K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat cv_D = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);

    cv::Mat mat_p(1, 1, CV_64FC2);
    mat_p.at<cv::Vec2d>(0, 0) = cv::Vec2d(p_p.x(), p_p.y());

    cv::Mat undistorted;
    cv::undistortPoints(mat_p, undistorted, cv_K, cv_D, cv::Mat(), cv_K);

    cv::Vec2d res = undistorted.at<cv::Vec2d>(0, 0);
    return Eigen::Vector2d(res[0], res[1]);
}

bool Camera::TriangulateDLT(const Eigen::Matrix4d &T_cw1, const Eigen::Matrix4d &T_cw2,
                            const Eigen::Vector2d &pt1_norm, const Eigen::Vector2d &pt2_norm,
                            Eigen::Vector3d &P_w)
{
    // 归一化坐标下，投影矩阵直接取 T_cw 的前 3x4 块 [R | t]
    Eigen::Matrix<double, 3, 4> P1 = T_cw1.block<3, 4>(0, 0);
    Eigen::Matrix<double, 3, 4> P2 = T_cw2.block<3, 4>(0, 0);

    Eigen::Vector4d p1_1 = P1.row(0), p1_2 = P1.row(1), p1_3 = P1.row(2);
    Eigen::Vector4d p2_1 = P2.row(0), p2_2 = P2.row(1), p2_3 = P2.row(2);

    // 构建方程组 A * X = 0
    Eigen::Matrix4d A;
    A.row(0) = pt1_norm.x() * p1_3 - p1_1;
    A.row(1) = pt1_norm.y() * p1_3 - p1_2;
    A.row(2) = pt2_norm.x() * p2_3 - p2_1;
    A.row(3) = pt2_norm.y() * p2_3 - p2_2;

    // SVD 求解
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_homo = svd.matrixV().col(3);

    if (std::abs(X_homo.w()) < 1e-6)
        return false;

    // 齐次坐标归一化
    P_w = X_homo.head<3>() / X_homo.w();

    // 正深度检查 (Check Positive Depth)
    Eigen::Vector3d P_c1 = T_cw1.block<3, 3>(0, 0) * P_w + T_cw1.block<3, 1>(0, 3);
    Eigen::Vector3d P_c2 = T_cw2.block<3, 3>(0, 0) * P_w + T_cw2.block<3, 1>(0, 3);

    return (P_c1.z() > 0.0 && P_c2.z() > 0.0);
}