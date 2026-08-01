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

    static cv::FileStorage g_fileConfig;
    static std::shared_ptr<Config> g_spConfig;

    // 相机内参及畸变参数 (Camera.*)
    static double g_dFx;
    static double g_dFy;
    static double g_dCx;
    static double g_dCy;
    static double g_dK1;
    static double g_dK2;
    static double g_dP1;
    static double g_dP2;

    // 相机基础设置
    static int g_nImageWidth;
    static int g_nImageHeight;
    static double g_dFps;
    static double g_dBf;       // baseline * fx
    static int g_nRGB;         // 0: BGR, 1: RGB
    static double g_dThDepth;  // 远近点阈值

    // ORB 提取器参数
    static int g_nORBnFeatures;
    static double g_dORBscaleFactor;
    static int g_nORBnLevels;
    static int g_nORBiniThFAST;
    static int g_nORBminThFAST;

    // 可视化 Viewer 参数
    static double g_dViewerKeyFrameSize;
    static double g_dViewerKeyFrameLineWidth;
    static double g_dViewerGraphLineWidth;
    static double g_dViewerPointSize;
    static double g_dViewerCameraSize;
    static double g_dViewerCameraLineWidth;
    static double g_dViewerPointX;
    static double g_dViewerPointY;
    static double g_dViewerPointZ;
    static double g_dViewerPointF;

    static bool setParameterFile(const std::string &strConfigFile);

    template <typename T>
    static T get(const std::string &strKey)
    {
        return T(Config::g_fileConfig[strKey]);
    }
};

#endif