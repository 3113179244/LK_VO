#include "Viewer.h"
#include "KeyFrame.h"
#include "MapPoint.h"

Viewer::Viewer(std::shared_ptr<Map> pMap) : mpMap(pMap), mbFinishRequested(false), mbFinished(false)
{
    // 初始化可视化参数
    mKeyFrameSize = 0.05f;
    mKeyFrameLineWidth = 1.0f;
    mPointSize = 2.0f;
    mCameraSize = 0.08f;
    mCameraLineWidth = 3.0f;
}

void Viewer::Run()
{
    // 1. 创建 Pangolin 窗口
    pangolin::CreateWindowAndBind("LK_VO: Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 2. 设置相机的投影和观察矩阵
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.7, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    // 3. 创建交互视图
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 白色背景

        d_cam.Activate(s_cam);

        // 渲染元素
        DrawMapPoints();
        DrawKeyFrames();

        pangolin::FinishFrame();

        // 检查终止信号
        std::unique_lock<std::mutex> lock(mMutexFinish);
        if (mbFinishRequested)
        {
            mbFinished = true;
            break;
        }
    }
}

void Viewer::DrawMapPoints()
{
    // 从 Map 中安全获取数据
    std::vector<std::shared_ptr<MapPoint>> vpMPs = mpMap->GetAllMapPoints();
    std::vector<std::shared_ptr<MapPoint>> vpActiveMPs = mpMap->GetActiveMapPoints();

    if (vpMPs.empty()) return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0); // 黑色表示全局历史点

    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        if (vpMPs[i]->isBad())
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }
    glEnd();

    // 绘制局部活跃点（红色以示区分）
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);
    for (size_t i = 0; i < vpActiveMPs.size(); i++)
    {
        if (vpActiveMPs[i]->isBad())
            continue;
        cv::Mat pos = vpActiveMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }
    glEnd();
}

void Viewer::DrawKeyFrames()
{
    // 渲染 KeyFrame 的位姿
    std::vector<std::shared_ptr<Frame>> vpKFs = mpMap->GetAllKeyFrames();

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        Eigen::Matrix4f Tcw = vpKFs[i]->GetPose();
        Eigen::Matrix4f Twc = Tcw.inverse(); // 从 Tcw 获取 Twc 以渲染相机在世界坐标系的姿态
    }
}

void Viewer::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}