#ifndef CAMERA_H
#define CAMERA_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>

class Config;

class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;

    Camera();

    static bool TriangulateDLT(const Eigen::Matrix4d &T_cw1, const Eigen::Matrix4d &T_cw2,
                               const Eigen::Vector2d &pt1_norm, const Eigen::Vector2d &pt2_norm,
                               Eigen::Vector3d &P_w);

    Eigen::Vector2d Camera2Pixel(const Eigen::Vector3d &P_c);
    Eigen::Vector3d Pixel2Camera(const Eigen::Vector2d &p_p);
    Eigen::Vector3d World2Camera(const Eigen::Vector3d &P_w, const Eigen::Matrix4d &T_cw);
    Eigen::Vector3d Camera2World(const Eigen::Vector3d &P_c, const Eigen::Matrix4d &T_wc);
    Eigen::Vector2d UndistortPoint(const Eigen::Vector2d &p_p);

public:
    const double fx, fy, cx, cy;
    const double k1, k2, p1, p2, k3;
    const double mBaseline;

    Eigen::Matrix3d K;
    Eigen::Matrix3d K_inv;
};

#endif // CAMERA_H