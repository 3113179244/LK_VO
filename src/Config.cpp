#include "Config.h"

std::shared_ptr<Config> Config::g_spConfig = nullptr;
cv::FileStorage Config::g_fileConfig;

// 初始化静态变量
double Config::g_dFx = 0.0;
double Config::g_dFy = 0.0;
double Config::g_dCx = 0.0;
double Config::g_dCy = 0.0;
double Config::g_dK1 = 0.0;
double Config::g_dK2 = 0.0;
double Config::g_dP1 = 0.0;
double Config::g_dP2 = 0.0;

int Config::g_nImageWidth = 0;
int Config::g_nImageHeight = 0;
double Config::g_dFps = 0.0;
double Config::g_dBf = 0.0;
int Config::g_nRGB = 0;
double Config::g_dThDepth = 0.0;

int Config::g_nORBnFeatures = 0;
double Config::g_dORBscaleFactor = 0.0;
int Config::g_nORBnLevels = 0;
int Config::g_nORBiniThFAST = 0;
int Config::g_nORBminThFAST = 0;

double Config::g_dViewerKeyFrameSize = 0.0;
double Config::g_dViewerKeyFrameLineWidth = 0.0;
double Config::g_dViewerGraphLineWidth = 0.0;
double Config::g_dViewerPointSize = 0.0;
double Config::g_dViewerCameraSize = 0.0;
double Config::g_dViewerCameraLineWidth = 0.0;
double Config::g_dViewerPointX = 0.0;
double Config::g_dViewerPointY = 0.0;
double Config::g_dViewerPointZ = 0.0;
double Config::g_dViewerPointF = 0.0;

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

    // 1. 读取 Camera Parameters
    g_dFx = get<double>("Camera.fx");
    g_dFy = get<double>("Camera.fy");
    g_dCx = get<double>("Camera.cx");
    g_dCy = get<double>("Camera.cy");

    g_dK1 = get<double>("Camera.k1");
    g_dK2 = get<double>("Camera.k2");
    g_dP1 = get<double>("Camera.p1");
    g_dP2 = get<double>("Camera.p2");

    g_nImageWidth  = get<int>("Camera.width");
    g_nImageHeight = get<int>("Camera.height");
    g_dFps         = get<double>("Camera.fps");
    g_dBf          = get<double>("Camera.bf");
    g_nRGB         = get<int>("Camera.RGB");
    g_dThDepth     = get<double>("ThDepth");

    // 2. 读取 ORB Parameters
    g_nORBnFeatures    = get<int>("ORBextractor.nFeatures");
    g_dORBscaleFactor  = get<double>("ORBextractor.scaleFactor");
    g_nORBnLevels      = get<int>("ORBextractor.nLevels");
    g_nORBiniThFAST    = get<int>("ORBextractor.iniThFAST");
    g_nORBminThFAST    = get<int>("ORBextractor.minThFAST");

    // 3. 读取 Viewer Parameters（做非空判断，避免配置文件未定义时出错）
    if (!g_fileConfig["Viewer.KeyFrameSize"].empty())
    {
        g_dViewerKeyFrameSize      = get<double>("Viewer.KeyFrameSize");
        g_dViewerKeyFrameLineWidth = get<double>("Viewer.KeyFrameLineWidth");
        g_dViewerGraphLineWidth    = get<double>("Viewer.GraphLineWidth");
        g_dViewerPointSize         = get<double>("Viewer.PointSize");
        g_dViewerCameraSize        = get<double>("Viewer.CameraSize");
        g_dViewerCameraLineWidth   = get<double>("Viewer.CameraLineWidth");
        g_dViewerPointX            = get<double>("Viewer.ViewpointX");
        g_dViewerPointY            = get<double>("Viewer.ViewpointY");
        g_dViewerPointZ            = get<double>("Viewer.ViewpointZ");
        g_dViewerPointF            = get<double>("Viewer.ViewpointF");
    }

    std::cout << "[Config] Successfully loaded KITTI configuration parameters!" << std::endl;
    return true;
}