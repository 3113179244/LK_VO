#ifndef CAMERA_H
#define CAMERA_H

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
    double baseline_ = 0;

    Camera() = default;
    Camera(double fx, double fy, double cx, double cy, double baseline = 0)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline) {}

    // 1. 世界坐标系 -> 相机坐标系: P_c = T_cw * P_w
    Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_cw);

    // 2. 相机坐标系 -> 世界坐标系: P_w = T_wc * P_c
    Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_cw);

    // 3. 相机坐标系 -> 2D 像素平面 (投影 Projection)
    Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c);

    // 4. 2D 像素平面 -> 相机归一化平面 z=1 (反投影 Back-Projection)
    Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, double depth = 1.0);

    // 5. 世界坐标系 -> 2D 像素平面 (一步到位)
    Eigen::Vector2d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_cw);

    // 6. 2D 像素平面 -> 世界坐标系 (需已知深度 depth)
    Eigen::Vector3d pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_cw, double depth = 1.0);
};

#endif // CAMERA_H