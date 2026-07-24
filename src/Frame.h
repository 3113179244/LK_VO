#ifndef FRAME_H
#define FRAME_H

#include <memory>
#include <mutex>
#include <atomic>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "Camera.h"

// 前向声明
struct Feature;

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame() = default;
    Frame(unsigned long id, double timestamp, const Sophus::SE3d &pose,
          const cv::Mat &image0, const cv::Mat &image1, Camera::Ptr camera);

    // 创建新帧
    static Frame::Ptr createFrame(const cv::Mat &image0,
                                  const cv::Mat &image1,
                                  Camera::Ptr camera,
                                  double timestamp = 0.0);

    unsigned long id() const;
    double timestamp() const;
    bool isKeyframe() const;
    void setKeyFrame();

    Sophus::SE3d getPose();
    void setPose(const Sophus::SE3d &pose);
    Eigen::Vector3d getCameraCenter();

    bool isInFrame(const Eigen::Vector3d &pt_world);
    Camera::Ptr getCamera() const;
    cv::Mat getImage() const;

private:
    cv::Mat left_img_;
    cv::Mat right_img_;
    Camera::Ptr camera_left_;
    Camera::Ptr camera_right_;
    static std::atomic<unsigned long> id_counter_; 

    unsigned long id_ = 0;   
    double timestamp_ = 0.0; 

    cv::Mat image_; 
    Camera::Ptr camera_ = nullptr;

    Sophus::SE3d pose_;    
    std::mutex pose_mutex_; 

    bool is_keyframe_ = false;
    unsigned long keyframe_id_ = 0;
};

#endif // FRAME_H