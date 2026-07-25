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

    static int g_nImu;
    static int g_nNumOfCam;
    static std::string g_strImage0Topic;
    static std::string g_strImage1Topic;
    static std::string g_strOutputPath;

    static int g_nImageWidth;
    static int g_nImageHeight;

    static Eigen::Matrix4d g_mBodyTCam0;
    static Eigen::Matrix4d g_mBodyTCam1;

    static int g_nMaxCnt;
    static int g_nMinDist;
    static int g_nFreq;
    static double g_dFThreshold;
    static int g_nShowTrack;
    static int g_nFlowBack;

    static double g_dMaxSolverTime;
    static int g_nMaxNumIterations;
    static double g_dKeyframeParallax;

    static double g_dAccN;
    static double g_dGyrN;
    static double g_dAccW;
    static double g_dGyrW;
    static double g_dGNorm;

    static double g_dFx0, g_dFy0, g_dCx0, g_dCy0;
    static double g_dK1_0, g_dK2_0, g_dP1_0, g_dP2_0;

    static double g_dFx1, g_dFy1, g_dCx1, g_dCy1;
    static double g_dK1_1, g_dK2_1, g_dP1_1, g_dP2_1;

    static bool setParameterFile(const std::string &strConfigFile);

    template <typename T>
    static T get(const std::string &strKey)
    {
        return T(Config::g_fileConfig[strKey]);
    }

private:
    static bool readCameraConfig(const std::string &strCamConfigFile,
                                 double &dFx, double &dFy, double &dCx, double &dCy,
                                 double &dK1, double &dK2, double &dP1, double &dP2);
    static Eigen::Matrix4d cvMat2Eigen(const cv::Mat &matCv);
};

#endif