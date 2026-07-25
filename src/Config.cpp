#include "Config.h"

std::shared_ptr<Config> Config::g_spConfig = nullptr;
cv::FileStorage Config::g_fileConfig;

int Config::g_nImu = 0;
int Config::g_nNumOfCam = 0;
std::string Config::g_strImage0Topic = "";
std::string Config::g_strImage1Topic = "";
std::string Config::g_strOutputPath = "";
int Config::g_nImageWidth = 0;
int Config::g_nImageHeight = 0;

Eigen::Matrix4d Config::g_mBodyTCam0 = Eigen::Matrix4d::Identity();
Eigen::Matrix4d Config::g_mBodyTCam1 = Eigen::Matrix4d::Identity();

int Config::g_nMaxCnt = 0;
int Config::g_nMinDist = 0;
int Config::g_nFreq = 0;
double Config::g_dFThreshold = 0.0;
int Config::g_nShowTrack = 0;
int Config::g_nFlowBack = 0;

double Config::g_dMaxSolverTime = 0.0;
int Config::g_nMaxNumIterations = 0;
double Config::g_dKeyframeParallax = 0.0;

double Config::g_dAccN = 0.0;
double Config::g_dGyrN = 0.0;
double Config::g_dAccW = 0.0;
double Config::g_dGyrW = 0.0;
double Config::g_dGNorm = 0.0;

double Config::g_dFx0 = 0, Config::g_dFy0 = 0, Config::g_dCx0 = 0, Config::g_dCy0 = 0;
double Config::g_dK1_0 = 0, Config::g_dK2_0 = 0, Config::g_dP1_0 = 0, Config::g_dP2_0 = 0;

double Config::g_dFx1 = 0, Config::g_dFy1 = 0, Config::g_dCx1 = 0, Config::g_dCy1 = 0;
double Config::g_dK1_1 = 0, Config::g_dK2_1 = 0, Config::g_dP1_1 = 0, Config::g_dP2_1 = 0;

Config::Config() {}

Config::~Config()
{
    if (g_fileConfig.isOpened())
    {
        g_fileConfig.release();
    }
}

bool Config::setParameterFile(const std::string &strConfigFile)
{
    if (g_spConfig == nullptr)
    {
        g_spConfig = std::shared_ptr<Config>(new Config());
    }

    g_fileConfig = cv::FileStorage(strConfigFile, cv::FileStorage::READ);
    if (!g_fileConfig.isOpened())
    {
        std::cerr << "[Config] Error: Cannot open config file: " << strConfigFile << std::endl;
        return false;
    }

    g_nImu = get<int>("imu");
    g_nNumOfCam = get<int>("num_of_cam");
    g_strImage0Topic = get<std::string>("image0_topic");
    g_strImage1Topic = get<std::string>("image1_topic");
    g_strOutputPath = get<std::string>("output_path");
    g_nImageWidth = get<int>("image_width");
    g_nImageHeight = get<int>("image_height");

    cv::Mat matCvT0, matCvT1;
    g_fileConfig["body_T_cam0"] >> matCvT0;
    g_fileConfig["body_T_cam1"] >> matCvT1;
    g_mBodyTCam0 = cvMat2Eigen(matCvT0);
    g_mBodyTCam1 = cvMat2Eigen(matCvT1);

    g_nMaxCnt = get<int>("max_cnt");
    g_nMinDist = get<int>("min_dist");
    g_nFreq = get<int>("freq");
    g_dFThreshold = get<double>("F_threshold");
    g_nShowTrack = get<int>("show_track");
    g_nFlowBack = get<int>("flow_back");

    g_dMaxSolverTime = get<double>("max_solver_time");
    g_nMaxNumIterations = get<int>("max_num_iterations");
    g_dKeyframeParallax = get<double>("keyframe_parallax");

    g_dAccN = get<double>("acc_n");
    g_dGyrN = get<double>("gyr_n");
    g_dAccW = get<double>("acc_w");
    g_dGyrW = get<double>("gyr_w");
    g_dGNorm = get<double>("g_norm");

    std::string strConfigDir = strConfigFile.substr(0, strConfigFile.find_last_of("/\\") + 1);
    std::string strCam0File = strConfigDir + get<std::string>("cam0_calib");
    std::string strCam1File = strConfigDir + get<std::string>("cam1_calib");

    readCameraConfig(strCam0File, g_dFx0, g_dFy0, g_dCx0, g_dCy0, g_dK1_0, g_dK2_0, g_dP1_0, g_dP2_0);
    if (g_nNumOfCam > 1)
    {
        readCameraConfig(strCam1File, g_dFx1, g_dFy1, g_dCx1, g_dCy1, g_dK1_1, g_dK2_1, g_dP1_1, g_dP2_1);
    }

    std::cout << "[Config] Successfully loaded configuration parameters!" << std::endl;
    return true;
}

bool Config::readCameraConfig(const std::string &strCamConfigFile,
                              double &dFx, double &dFy, double &dCx, double &dCy,
                              double &dK1, double &dK2, double &dP1, double &dP2)
{
    cv::FileStorage fileCam(strCamConfigFile, cv::FileStorage::READ);
    if (!fileCam.isOpened())
    {
        std::cerr << "[Config] Error: Cannot open camera config file: " << strCamConfigFile << std::endl;
        return false;
    }

    cv::FileNode nodeProj = fileCam["projection_parameters"];
    dFx = static_cast<double>(nodeProj["fx"]);
    dFy = static_cast<double>(nodeProj["fy"]);
    dCx = static_cast<double>(nodeProj["cx"]);
    dCy = static_cast<double>(nodeProj["cy"]);

    cv::FileNode nodeDist = fileCam["distortion_parameters"];
    dK1 = static_cast<double>(nodeDist["k1"]);
    dK2 = static_cast<double>(nodeDist["k2"]);
    dP1 = static_cast<double>(nodeDist["p1"]);
    dP2 = static_cast<double>(nodeDist["p2"]);

    fileCam.release();
    return true;
}

Eigen::Matrix4d Config::cvMat2Eigen(const cv::Mat &matCv)
{
    Eigen::Matrix4d mEigen = Eigen::Matrix4d::Identity();
    if (matCv.rows == 4 && matCv.cols == 4)
    {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                mEigen(i, j) = matCv.at<double>(i, j);
            }
        }
    }
    return mEigen;
}