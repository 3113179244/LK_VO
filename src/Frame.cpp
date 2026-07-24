#include "Frame.h"

std::atomic<unsigned long> Frame::id_counter_{0};

Frame::Frame(unsigned long id, double timestamp, const Sophus::SE3d &pose,
             const cv::Mat &image0, const cv::Mat &image1, Camera::Ptr camera)
    : id_(id), timestamp_(timestamp), pose_(pose), left_img_(image0), right_img_(image1), camera_(camera) {}

Frame::Ptr Frame::createFrame(const cv::Mat &image0,
                              const cv::Mat &image1,
                              Camera::Ptr camera,
                              double timestamp)
{
    return std::make_shared<Frame>(id_counter_++, timestamp, Sophus::SE3d(), image0, image1, camera);
}

unsigned long Frame::id() const
{
    return id_;
}

double Frame::timestamp() const
{
    return timestamp_;
}

bool Frame::isKeyframe() const
{
    return is_keyframe_;
}

void Frame::setKeyFrame()
{
    static unsigned long keyframe_counter = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_counter++;
}

Sophus::SE3d Frame::getPose()
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_;
}

void Frame::setPose(const Sophus::SE3d &pose)
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    pose_ = pose;
}

Eigen::Vector3d Frame::getCameraCenter()
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_.inverse().translation();
}

bool Frame::isInFrame(const Eigen::Vector3d &pt_world)
{
    Sophus::SE3d T_cw;
    {
        std::unique_lock<std::mutex> lock(pose_mutex_);
        T_cw = pose_;
    }

    Eigen::Vector3d p_c = camera_->world2camera(pt_world, T_cw);

    if (p_c[2] < 0.1)
        return false;

    Eigen::Vector2d pixel = camera_->camera2pixel(p_c);
    int border = 1;
    return pixel[0] >= border && pixel[1] >= border &&
           pixel[0] < (image_.cols - border) &&
           pixel[1] < (image_.rows - border);
}

Camera::Ptr Frame::getCamera() const
{
    return camera_;
}

cv::Mat Frame::getImage() const
{
    return image_;
}