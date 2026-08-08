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
    std::string strConfigFile = "/home/wzj/Stereo_ORB_VO/config/KITTI04-12.yaml";
    std::string strSequenceDir = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/05";
    std::string strVocFile = "/home/wzj/DBow3/orbvoc.dbow3";
    // 解析命令行参数
    if (argc >= 2)
    {
        strConfigFile = argv[1];
    }
    if (argc >= 3)
    {
        strSequenceDir = argv[2];
    }
    if (argc >= 4) 
    {
        strVocFile = argv[3];
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
    System SLAM(strConfigFile, strVocFile, System::STEREO, true);
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
            cv::Mat image0 = cv::imread(strLeftImgPath, cv::IMREAD_GRAYSCALE);
            cv::Mat image1 = cv::imread(strRightImgPath, cv::IMREAD_GRAYSCALE);

            if (image0.empty() || image1.empty())
            {
                std::cerr << "错误: 无法读取图像: " << strFilename << std::endl;
                break;
            }
            double dTimestamp = vdTimestamps[nFrameId];
            Eigen::Matrix4f Tcw = SLAM.TrackStereo(image0, image1, dTimestamp);
            if (!image0.empty() && !image1.empty())
            {
                cv::Mat imDraw = SLAM.DrawFrame();
                if (!imDraw.empty())
                {
                    cv::imshow("ORB-SLAM2 Frame Drawer", imDraw);
                }
            }
            nFrameId++;
        }
        int nWaitTime = bIsPaused ? 10 : 20;
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
    }

    cv::destroyAllWindows();

    // 保存逐帧轨迹（KITTI 格式，供 evo 评估；每行 12 个数，无时间戳）
    // 方案C：改为保存「每一帧」的轨迹而非「关键帧」轨迹，使行数与图像帧数一致，
    //        从而与 KITTI ground truth（如 04.txt）逐行对齐评估，
    //        解决 evo_ape 报 "data matrices must have the same shape" 的问题。
    std::string strTrajDir = "/home/wzj/output";
    cv::utils::fs::createDirectories(strTrajDir);  // 确保输出目录存在（不存在则创建）
    std::string strTrajFile = strTrajDir + "/trajectory.txt";
    SLAM.SaveFrameTrajectoryKITTI(strTrajFile);

    return 0;
}