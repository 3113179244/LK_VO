#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

class Camera {
public:
    typedef std::shared_ptr<Camera> Ptr;

    // 构造函数：传入内参、畸变参数和基线
    Camera(float fx, float fy, float cx, float cy,
           float k1, float k2, float p1, float p2, float k3,
           float baseline);

    // ==========================================
    // 1. 核心坐标转换 (Projection / Unprojection)
    // ==========================================
    
    // 相机坐标系 (3D) -> 像素坐标系 (2D)
    cv::Point2f Camera2Pixel(const cv::Mat& P_c);
    cv::Point2f Camera2Pixel(const cv::Point3f& P_c);

    // 像素坐标系 (2D) -> 相机坐标系归一化平面 (3D, Z=1)
    // 这是将像素反推回射线的关键步骤
    cv::Point3f Pixel2Camera(const cv::Point2f& p_p);

    // ==========================================
    // 2. 世界坐标系与相机坐标系的转换
    // ==========================================
    
    // 世界坐标系 (3D) -> 相机坐标系 (3D)
    // T_cw: 从世界到相机的变换矩阵 (4x4)
    cv::Point3f World2Camera(const cv::Point3f& P_w, const cv::Mat& T_cw);

    // 相机坐标系 (3D) -> 世界坐标系 (3D)
    // T_wc: 从相机到世界的变换矩阵 (4x4)，通常是 T_cw 的逆
    cv::Point3f Camera2World(const cv::Point3f& P_c, const cv::Mat& T_wc);

    // ==========================================
    // 3. 畸变处理 (Undistortion)
    // ==========================================
    
    // 对单个特征点进行去畸变处理
    // 注意：如果图像在送入 SLAM 前已经整体做过校正和去畸变，此函数可以跳过
    cv::Point2f UndistortPoint(const cv::Point2f& p_p);

public:
    // ==========================================
    // 相机参数 (只读)
    // ==========================================
    
    // 内参 (Intrinsics)
    const float fx, fy, cx, cy;
    
    // 内参矩阵 K
    cv::Mat K; 
    
    // 内参矩阵的逆 K_inv (提前计算好，加速 Pixel2Camera 过程)
    cv::Mat K_inv; 
    
    // 畸变参数 (Distortion Coefficients)
    const float k1, k2, p1, p2, k3;
    
    // 双目基线长度 (单位通常是米)
    const float mBaseline;
};

#endif // CAMERA_H