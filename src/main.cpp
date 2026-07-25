#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "System.h"
#include "Config.h"

int main(int argc, char **argv)
{
    // 设置默认路径
    std::string strConfigFile = "/home/wzj/LK_VO/config/kitti_config04-12.yaml";
    std::string strSequenceDir = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/05";

    // 解析命令行参数
    if (argc >= 2)
    {
        strConfigFile = argv[1];
    }
    if (argc >= 3)
    {
        strSequenceDir = argv[2];
    }

    // 确保序列路径末尾有 '/'
    if (!strSequenceDir.empty() && strSequenceDir.back() != '/' && strSequenceDir.back() != '\\')
    {
        strSequenceDir += "/";
    }

    // 更新完 strSequenceDir 之后，再拼接子目录路径！
    std::string strLeftDir = strSequenceDir + "image_0/";
    std::string strRightDir = strSequenceDir + "image_1/";
    std::string strTimesPath = strSequenceDir + "times.txt";

    // 初始化 VO 系统
    System SLAM(strConfigFile, System::STEREO, true);

    // 检查路径
    if (!cv::utils::fs::exists(strLeftDir) || !cv::utils::fs::exists(strRightDir))
    {
        std::cerr << "错误: 找不到路径 " << strLeftDir << " 或 " << strRightDir << " ！" << std::endl;
        return -1;
    }
    // 加载时间戳文件
    std::vector<double> vdTimestamps;
    std::ifstream fileTimes(strTimesPath);
    if (!fileTimes.is_open())
    {
        std::cerr << "错误: 无法打开时间戳文件 " << strTimesPath << std::endl;
        return -1;
    }

    double dTimestamp = 0.0;
    while (fileTimes >> dTimestamp)
    {
        vdTimestamps.push_back(dTimestamp);
    }
    fileTimes.close();

    std::cout << "成功加载 " << vdTimestamps.size() << " 个时间戳。" << std::endl;
    std::cout << "  - 空格键 (Space): 暂停/恢复播放" << std::endl;
    std::cout << "  - Q 键           : 恢复播放" << std::endl;
    std::cout << "  - ESC 键         : 退出程序" << std::endl;

    int nFrameId = 0;
    bool bIsPaused = false;

    // 循环处理每一帧
    while (true)
    {
        std::stringstream ssFilename;
        ssFilename << std::setw(6) << std::setfill('0') << nFrameId << ".png";
        std::string strFilename = ssFilename.str();

        std::string strLeftImgPath = strLeftDir + strFilename;
        std::string strRightImgPath = strRightDir + strFilename;

        // 检查文件是否存在以及是否到达末尾
        if (!cv::utils::fs::exists(strLeftImgPath) || !cv::utils::fs::exists(strRightImgPath) || nFrameId >= static_cast<int>(vdTimestamps.size()))
        {
            std::cout << "\n已到达序列末尾，播放结束。共处理 " << nFrameId << " 帧。" << std::endl;
            break;
        }

        if (!bIsPaused)
        {
            cv::Mat matImgLeft = cv::imread(strLeftImgPath, cv::IMREAD_GRAYSCALE);
            cv::Mat matImgRight = cv::imread(strRightImgPath, cv::IMREAD_GRAYSCALE);

            if (matImgLeft.empty() || matImgRight.empty())
            {
                std::cerr << "错误: 无法读取图像: " << strFilename << std::endl;
                break;
            }

            double dCurrentTimestamp = vdTimestamps[nFrameId];

            // 将图像和时间戳传给 VO 系统进行双目跟踪和位姿估计
            Eigen::Matrix4f mTcw = SLAM.TrackStereo(matImgLeft, matImgRight, dCurrentTimestamp);

            // 绘制UI信息
            std::stringstream ssText;
            ssText << "Frame: " << nFrameId << " | Time: " << std::fixed << std::setprecision(4) << dCurrentTimestamp << "s";

            cv::putText(matImgLeft, ssText.str(), cv::Point(30, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);

            std::string strStatusText = bIsPaused ? "PAUSED" : "PLAYING";
            cv::putText(matImgLeft, strStatusText, cv::Point(matImgLeft.cols - 160, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);

            cv::imshow("KITTI VO (Left)", matImgLeft);
        }

        int nWaitTime = bIsPaused ? 0 : 20;
        char cKey = static_cast<char>(cv::waitKey(nWaitTime));

        if (cKey == 27) // ESC
        {
            std::cout << "\n按下 ESC，退出程序。" << std::endl;
            break;
        }
        else if (cKey == ' ') // Space
        {
            bIsPaused = !bIsPaused;
            if (bIsPaused)
                std::cout << "\r[状态] 已暂停播放 (按 Space/Q 键继续)... " << std::flush;
            else
                std::cout << "\r[状态] 恢复播放...                      " << std::flush;
        }
        else if (cKey == 'q' || cKey == 'Q')
        {
            if (bIsPaused)
            {
                bIsPaused = false;
                std::cout << "\r[状态] 恢复播放...                      " << std::flush;
            }
        }

        if (!bIsPaused)
        {
            nFrameId++;
        }
    }

    // 7. 安全关闭系统线程
    SLAM.Shutdown();
    cv::destroyAllWindows();

    return 0;
}