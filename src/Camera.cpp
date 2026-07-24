#include "Camera.h"

Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_cw)
{
    return T_cw * p_w;
}

Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_cw)
{
    return T_cw.inverse() * p_c;
}

Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c)
{
    return Eigen::Vector2d(
        fx_ * p_c[0] / p_c[2] + cx_,
        fy_ * p_c[1] / p_c[2] + cy_);
}

Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth)
{
    return Eigen::Vector3d(
        (p_p[0] - cx_) * depth / fx_,
        (p_p[1] - cy_) * depth / fy_,
        depth);
}

Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_cw)
{
    return camera2pixel(world2camera(p_w, T_cw));
}

Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_cw, double depth)
{
    return camera2world(pixel2camera(p_p, depth), T_cw);
}