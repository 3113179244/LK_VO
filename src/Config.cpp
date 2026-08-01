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

    // 1. 读取 Camera Parameters (使用安全转换)
    g_fileConfig["Camera.fx"] >> g_dFx;
    g_fileConfig["Camera.fy"] >> g_dFy;
    g_fileConfig["Camera.cx"] >> g_dCx;
    g_fileConfig["Camera.cy"] >> g_dCy;

    g_fileConfig["Camera.k1"] >> g_dK1;
    g_fileConfig["Camera.k2"] >> g_dK2;
    g_fileConfig["Camera.p1"] >> g_dP1;
    g_fileConfig["Camera.p2"] >> g_dP2;

    g_fileConfig["Camera.width"]  >> g_nImageWidth;
    g_fileConfig["Camera.height"] >> g_nImageHeight;
    g_fileConfig["Camera.fps"]    >> g_dFps;
    g_fileConfig["Camera.bf"]     >> g_dBf;
    g_fileConfig["Camera.RGB"]    >> g_nRGB;
    g_fileConfig["ThDepth"]       >> g_dThDepth;

    // 2. 读取 ORB Parameters
    g_fileConfig["ORBextractor.nFeatures"]   >> g_nORBnFeatures;
    g_fileConfig["ORBextractor.scaleFactor"] >> g_dORBscaleFactor;
    g_fileConfig["ORBextractor.nLevels"]     >> g_nORBnLevels;
    g_fileConfig["ORBextractor.iniThFAST"]   >> g_nORBiniThFAST;
    g_fileConfig["ORBextractor.minThFAST"]   >> g_nORBminThFAST;

    // 3. 读取 Viewer Parameters
    if (!g_fileConfig["Viewer.KeyFrameSize"].empty())
    {
        g_fileConfig["Viewer.KeyFrameSize"]      >> g_dViewerKeyFrameSize;
        g_fileConfig["Viewer.KeyFrameLineWidth"] >> g_dViewerKeyFrameLineWidth;
        g_fileConfig["Viewer.GraphLineWidth"]    >> g_dViewerGraphLineWidth;
        g_fileConfig["Viewer.PointSize"]         >> g_dViewerPointSize;
        g_fileConfig["Viewer.CameraSize"]        >> g_dViewerCameraSize;
        g_fileConfig["Viewer.CameraLineWidth"]   >> g_dViewerCameraLineWidth;
        g_fileConfig["Viewer.ViewpointX"]        >> g_dViewerPointX;
        g_fileConfig["Viewer.ViewpointY"]        >> g_dViewerPointY;
        g_fileConfig["Viewer.ViewpointZ"]        >> g_dViewerPointZ;
        g_fileConfig["Viewer.ViewpointF"]        >> g_dViewerPointF;
    }

    std::cout << "[Config] Successfully loaded KITTI configuration parameters!" << std::endl;
    return true;
}