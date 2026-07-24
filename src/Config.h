#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class Config
{
public:
    Config();
    ~Config();
    static cv::FileStorage file_;
    static std::shared_ptr<Config> config_;
    static int imu;
    static int num_of_cam;
    static std::string image0_topic;
    static std::string image1_topic;
    static std::string output_path;

    static int image_width;
    static int image_height;

    static Eigen::Matrix4d body_T_cam0;
    static Eigen::Matrix4d body_T_cam1;

    static int max_cnt;
    static int min_dist;
    static int freq;
    static double F_threshold;
    static int show_track;
    static int flow_back;

    static double max_solver_time;
    static int max_num_iterations;
    static double keyframe_parallax;

    static double acc_n;
    static double gyr_n;
    static double acc_w;
    static double gyr_w;
    static double g_norm;

    static double fx0, fy0, cx0, cy0;
    static double k1_0, k2_0, p1_0, p2_0;

    static double fx1, fy1, cx1, cy1;
    static double k1_1, k2_1, p1_1, p2_1;
    static bool setParameterFile(const std::string &config_file);
    template <typename T>
    static T get(const std::string &key)
    {
        return T(Config::file_[key]);
    }

private:
    static bool readCameraConfig(const std::string &cam_config_file,
                                 double &fx, double &fy, double &cx, double &cy,
                                 double &k1, double &k2, double &p1, double &p2);
    static Eigen::Matrix4d cvMat2Eigen(const cv::Mat &cvMat);
};

#endif 