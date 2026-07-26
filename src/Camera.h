#ifndef CAMERA_H
#define CAMERA_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>

class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;

    // 构造函数：传入内参、畸变参数和基线
    Camera(float fx, float fy, float cx, float cy,
           float k1, float k2, float p1, float p2, float k3,
           float baseline);
    /**
     * @brief 使用 DLT 算法进行双视图三角化
     * @param T_cw1 相机1的位姿矩阵 T_cw (4x4)
     * @param T_cw2 相机2的位姿矩阵 T_cw (4x4)
     * @param pt1 相机1归一化平面上的点 (x1, y1, 1) 或像素点 (u1, v1)
     * @param pt2 相机2归一化平面上的点 (x2, y2, 1) 或像素点 (u2, v2)
     * @param P_w 输出求解出来的 3D 世界坐标点
     * @param isNormalized true 表示输入点为归一化平面坐标 (x,y)，false 表示输入点为像素坐标 (u,v)
     * @return true 三角化成功（深度正向）；false 三角化失败（在相机后方/深度非正）
     */
    static bool TriangulateDLT(const Eigen::Matrix4f &T_cw1, const Eigen::Matrix4f &T_cw2,
                           const Eigen::Vector2f &pt1_norm, const Eigen::Vector2f &pt2_norm,
                           Eigen::Vector3f &P_w);

    // 相机坐标系 (3D) -> 像素坐标系 (2D)
    Eigen::Vector2f Camera2Pixel(const Eigen::Vector3f &P_c);

    // 像素坐标系 (2D) -> 相机坐标系归一化平面 (3D, Z=1)
    Eigen::Vector3f Pixel2Camera(const Eigen::Vector2f &p_p);

    // 世界坐标系 (3D) -> 相机坐标系 (3D)
    // T_cw: 从世界到相机的变换矩阵 (4x4)
    Eigen::Vector3f World2Camera(const Eigen::Vector3f &P_w, const Eigen::Matrix4f &T_cw);

    // 相机坐标系 (3D) -> 世界坐标系 (3D)
    // T_wc: 从相机到世界的变换矩阵 (4x4)
    Eigen::Vector3f Camera2World(const Eigen::Vector3f &P_c, const Eigen::Matrix4f &T_wc);

    // 对单个特征点进行去畸变处理
    Eigen::Vector2f UndistortPoint(const Eigen::Vector2f &p_p);

public:
    // 内参 (Intrinsics)
    const float fx, fy, cx, cy;

    // 内参矩阵 K (3x3)
    Eigen::Matrix3f K;

    // 内参矩阵的逆 K_inv (3x3)
    Eigen::Matrix3f K_inv;

    // 畸变参数 (Distortion Coefficients)
    const float k1, k2, p1, p2, k3;

    // 双目基线长度 (单位通常是米)
    const float mBaseline;
};

#endif // CAMERA_H