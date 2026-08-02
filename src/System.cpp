#include "System.h"
#include "Config.h"
#include "Map.h"
#include "Tracker.h"
#include "Viewer.h"
#include "FrameDrawer.h"
System::System(const std::string &strConfigFile, const eSensor sensor, const bool bUseViewer)
    : mSensor(sensor), mpViewerThread(nullptr)
{
    std::cout << "Starting ORB-SLAM2 Stereo System..." << std::endl;

    // 加载参数配置文件
    if (!Config::setParameterFile(strConfigFile))
    {
        std::cerr << "[System] Failed to load config file: " << strConfigFile << std::endl;
        return;
    }

    // 初始化全局地图 Map
    mpMap = std::make_shared<Map>();
    mpFrameDrawer = std::make_shared<FrameDrawer>(mpMap.get());
    // 初始化前端 Tracker
    mpTracker = std::make_shared<Tracker>(this, mpMap, mSensor);
    mpTracker->SetFrameDrawer(mpFrameDrawer);
    // 如果开启 Viewer 模式，启动 Pangolin 可视化线程
    if (bUseViewer)
    {
        mpViewer = std::make_shared<Viewer>(this, mpMap);
        mpViewerThread = new std::thread(&Viewer::Run, mpViewer.get());
        mpTracker->SetViewer(mpViewer);
    }
}

System::~System()
{
    Shutdown();
}

cv::Mat System::DrawFrame()
{
    if (mpFrameDrawer)
        return mpFrameDrawer->DrawFrame();
    return cv::Mat();
}

Eigen::Matrix4f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if (mSensor != STEREO)
    {
        std::cerr << "Error: System initialized for non-stereo tracking!" << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    // 彩色/灰度检查
    cv::Mat imLeftGray = imLeft;
    cv::Mat imRightGray = imRight;

    if (imLeft.channels() == 3)
    {
        cv::cvtColor(imLeft, imLeftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imRight, imRightGray, cv::COLOR_BGR2GRAY);
    }

    // 调用 Tracker 执行跟踪主流程
    Eigen::Matrix4f Tcw = mpTracker->GrabImageStereo(imLeftGray, imRightGray, timestamp);

    return Tcw;
}

void System::Shutdown()
{
    if (mpViewerThread)
    {
        if (mpViewer)
            mpViewer->RequestStop();
        mpViewerThread->join();
        delete mpViewerThread;
        mpViewerThread = nullptr;
    }
}

void System::Reset()
{
    std::unique_lock<std::mutex> lock(mMutexMode);
    if (mpTracker)
        mpTracker->Reset();
    if (mpMap)
        mpMap->Clear();
}