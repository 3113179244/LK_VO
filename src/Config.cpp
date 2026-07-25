#include "Config.h"

std::shared_ptr<Config> Config::config_ = nullptr;
cv::FileStorage Config::file_;

int Config::imu = 0;
int Config::num_of_cam = 0;
std::string Config::image0_topic = "";
std::string Config::image1_topic = "";
std::string Config::output_path = "";
int Config::image_width = 0;
int Config::image_height = 0;

Eigen::Matrix4d Config::body_T_cam0 = Eigen::Matrix4d::Identity();
Eigen::Matrix4d Config::body_T_cam1 = Eigen::Matrix4d::Identity();

int Config::max_cnt = 0;
int Config::min_dist = 0;
int Config::freq = 0;
double Config::F_threshold = 0.0;
int Config::show_track = 0;
int Config::flow_back = 0;

double Config::max_solver_time = 0.0;
int Config::max_num_iterations = 0;
double Config::keyframe_parallax = 0.0;

double Config::acc_n = 0.0;
double Config::gyr_n = 0.0;
double Config::acc_w = 0.0;
double Config::gyr_w = 0.0;
double Config::g_norm = 0.0;

double Config::fx0 = 0, Config::fy0 = 0, Config::cx0 = 0, Config::cy0 = 0;
double Config::k1_0 = 0, Config::k2_0 = 0, Config::p1_0 = 0, Config::p2_0 = 0;

double Config::fx1 = 0, Config::fy1 = 0, Config::cx1 = 0, Config::cy1 = 0;
double Config::k1_1 = 0, Config::k2_1 = 0, Config::p1_1 = 0, Config::p2_1 = 0;
Config::Config(){}
Config::~Config()
{
    if (file_.isOpened())
    {
        file_.release();
    }
}

bool Config::setParameterFile(const std::string &config_file)
{
    if (config_ == nullptr)
    {
        config_ = std::shared_ptr<Config>(new Config());
    }

    file_ = cv::FileStorage(config_file, cv::FileStorage::READ);
    if (!file_.isOpened())
    {
        std::cerr << "[Config] Error: Cannot open config file: " << config_file << std::endl;
        return false;
    }

    imu = get<int>("imu");
    num_of_cam = get<int>("num_of_cam");
    image0_topic = get<std::string>("image0_topic");
    image1_topic = get<std::string>("image1_topic");
    output_path = get<std::string>("output_path");
    image_width = get<int>("image_width");
    image_height = get<int>("image_height");

    cv::Mat cv_T0, cv_T1;
    file_["body_T_cam0"] >> cv_T0;
    file_["body_T_cam1"] >> cv_T1;
    body_T_cam0 = cvMat2Eigen(cv_T0);
    body_T_cam1 = cvMat2Eigen(cv_T1);

    max_cnt = get<int>("max_cnt");
    min_dist = get<int>("min_dist");
    freq = get<int>("freq");
    F_threshold = get<double>("F_threshold");
    show_track = get<int>("show_track");
    flow_back = get<int>("flow_back");

    max_solver_time = get<double>("max_solver_time");
    max_num_iterations = get<int>("max_num_iterations");
    keyframe_parallax = get<double>("keyframe_parallax");

    acc_n = get<double>("acc_n");
    gyr_n = get<double>("gyr_n");
    acc_w = get<double>("acc_w");
    gyr_w = get<double>("gyr_w");
    g_norm = get<double>("g_norm");

    std::string config_dir = config_file.substr(0, config_file.find_last_of("/\\") + 1);
    std::string cam0_file = config_dir + get<std::string>("cam0_calib");
    std::string cam1_file = config_dir + get<std::string>("cam1_calib");

    readCameraConfig(cam0_file, fx0, fy0, cx0, cy0, k1_0, k2_0, p1_0, p2_0);
    if (num_of_cam > 1)
    {
        readCameraConfig(cam1_file, fx1, fy1, cx1, cy1, k1_1, k2_1, p1_1, p2_1);
    }

    std::cout << "[Config] Successfully loaded configuration parameters!" << std::endl;
    return true;
}

bool Config::readCameraConfig(const std::string &cam_config_file,
                              double &fx, double &fy, double &cx, double &cy,
                              double &k1, double &k2, double &p1, double &p2)
{
    cv::FileStorage cam_file(cam_config_file, cv::FileStorage::READ);
    if (!cam_file.isOpened())
    {
        std::cerr << "[Config] Error: Cannot open camera config file: " << cam_config_file << std::endl;
        return false;
    }

    cv::FileNode proj_node = cam_file["projection_parameters"];
    fx = static_cast<double>(proj_node["fx"]);
    fy = static_cast<double>(proj_node["fy"]);
    cx = static_cast<double>(proj_node["cx"]);
    cy = static_cast<double>(proj_node["cy"]);

    cv::FileNode dist_node = cam_file["distortion_parameters"];
    k1 = static_cast<double>(dist_node["k1"]);
    k2 = static_cast<double>(dist_node["k2"]);
    p1 = static_cast<double>(dist_node["p1"]);
    p2 = static_cast<double>(dist_node["p2"]);

    cam_file.release();
    return true;
}

Eigen::Matrix4d Config::cvMat2Eigen(const cv::Mat &cvMat)
{
    Eigen::Matrix4d eigenMat = Eigen::Matrix4d::Identity();
    if (cvMat.rows == 4 && cvMat.cols == 4)
    {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                eigenMat(i, j) = cvMat.at<double>(i, j);
            }
        }
    }
    return eigenMat;
}