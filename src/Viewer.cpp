#include "Viewer.h"
#include "Tracker.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"

#include <opencv2/highgui/highgui.hpp>
#include <GL/gl.h>
#include <chrono>

Viewer::Viewer(System *pSystem, std::shared_ptr<Map> pMap, std::shared_ptr<FrameDrawer> pFrameDrawer)
    : mpSystem(pSystem),
      mpMap(pMap),
      mpFrameDrawer(pFrameDrawer),
      mpTracker(nullptr),
      mCameraSize(0.15f),
      mCameraLineWidth(2.0f),
      mPointSize(2.0f),
      mKeyFrameSize(0.08f),
      mKeyFrameLineWidth(1.0f),
      mGraphLineWidth(0.9f),
      mbStopRequested(false),
      mbStopped(false),
      mbFinishRequested(false),
      mbFinished(false)
{
    mFPS = 30.0;
    mT = 1.0 / mFPS;
    mCameraPose = Eigen::Matrix4f::Identity();
}

Viewer::~Viewer() {}

void Viewer::UpdateCurrentCameraPose(const Eigen::Matrix4f &Tcw)
{
    SetCurrentCameraPose(Tcw);
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    // 1. 初始化 Pangolin 窗口
    pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 2. GUI 面板
    pangolin::CreatePanel("gui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("gui.Follow Camera", true, true);
    pangolin::Var<bool> menuShowTrajectory("gui.Show Trajectory", true, true);
    pangolin::Var<bool> menuShowKeyFrames("gui.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("gui.Show Graph", true, true);
    pangolin::Var<bool> menuShowPoints("gui.Show Points", true, true);
    pangolin::Var<bool> menuReset("gui.Reset", false, false);

    // 3. 观察相机设置
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.7, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    if (mpFrameDrawer)
    {
        cv::namedWindow("ORB-SLAM2: Frame Viewer");
    }

    bool bFollow = true;

    while (1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 获取当前帧经转换后的 OpenGL 相机位姿矩阵 Twc
        GetCurrentOpenGLCameraMatrix(Twc);

        // 视角跟随逻辑：只有当菜单勾选 Follow Camera 时，才调用 Follow(Twc)
        if (menuFollowCamera)
        {
            s_cam.Follow(Twc);
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 白色背景

        // 渲染 3D 元素
        DrawCurrentCamera(Twc);

        if (menuShowTrajectory)
            DrawTrajectory();

        if (menuShowKeyFrames || menuShowGraph)
            DrawKeyFrames(menuShowKeyFrames, menuShowGraph);

        if (menuShowPoints)
            DrawMapPoints();

        pangolin::FinishFrame();

        // 渲染 2D 特征图
        if (mpFrameDrawer)
        {
            cv::Mat im = mpFrameDrawer->DrawFrame();
            if (!im.empty())
            {
                cv::imshow("ORB-SLAM2: Frame Viewer", im);
                cv::waitKey(mT * 1000);
            }
        }

        if (pangolin::Pushed(menuReset))
        {
            if (mpTracker)
                mpTracker->Reset();
        }

        if (pangolin::ShouldQuit())
            break;

        {
            std::unique_lock<std::mutex> lock(mMutexFinish);
            if (mbFinishRequested)
                break;
        }
    }

    if (mpFrameDrawer)
    {
        cv::destroyWindow("ORB-SLAM2: Frame Viewer");
    }

    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        mbFinished = true;
    }
}

// 绘制运动轨迹（连线路径）
void Viewer::DrawTrajectory()
{
    if (!mpMap) return;

    const std::vector<KeyFrame *> &vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.size() < 2) return;

    // 1. 只保留有效关键帧，并按时间 ID 排序，确保连线按时间先后
    std::vector<KeyFrame *> vSorted;
    vSorted.reserve(vpKFs.size());
    for (KeyFrame *pKF : vpKFs) {
        if (!pKF || pKF->mbBad)      // 跳过坏关键帧
            continue;
        vSorted.push_back(pKF);
    }
    std::sort(vSorted.begin(), vSorted.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });
    if (vSorted.size() < 2) return;

    // 2. 用 GL_LINE_STRIP 按时间顺序连线
    glLineWidth(2.0f);
    glColor3f(0.0f, 1.0f, 0.0f); // 绿色轨迹线
    glBegin(GL_LINE_STRIP);
    for (KeyFrame *pKF : vSorted) {
        Eigen::Vector3f Ow = pKF->GetCameraCenter();
        glVertex3f(Ow.x(), Ow.y(), Ow.z());
    }
    glEnd();
}

void Viewer::DrawMapPoints()
{
    if (!mpMap)
        return;

    const std::vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
    if (vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 0.0f, 0.0f);

    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        MapPoint *pMP = vpMPs[i];
        if (!pMP || pMP->isBad())
            continue;

        Eigen::Vector3f pos = pMP->GetWorldPos();
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    glEnd();
}

void Viewer::DrawKeyFrames(bool bDrawKF, bool bDrawGraph)
{
    if (!mpMap)
        return;

    const std::vector<KeyFrame *> &vpKFs = mpMap->GetAllKeyFrames();

    if (bDrawKF)
    {
        const float w = mKeyFrameSize;
        const float h = w * 0.75f;
        const float z = w * 0.6f;

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (!pKF)
                continue;

            Eigen::Matrix4f Twc = pKF->GetPoseInverse();

            glPushMatrix();
            glMultMatrixf(Twc.data());

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f, 0.0f, 1.0f); // 关键帧：蓝色

            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(w, h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, h, z);

            glVertex3f(w, h, z);
            glVertex3f(w, -h, z);
            glVertex3f(w, -h, z);
            glVertex3f(-w, -h, z);
            glVertex3f(-w, -h, z);
            glVertex3f(-w, h, z);
            glVertex3f(-w, h, z);
            glVertex3f(w, h, z);
            glEnd();

            glPopMatrix();
        }
    }

    if (bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f, 0.7f, 0.7f, 0.5f); // 共视图：青色/半透明

        glBegin(GL_LINES);
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (!pKF)
                continue;

            Eigen::Vector3f Ow = pKF->GetCameraCenter();
            const std::vector<KeyFrame *> vCovKFs = pKF->GetBestCovisibilityKeyFrames(10);

            for (KeyFrame *pCovKF : vCovKFs)
            {
                if (!pCovKF || pCovKF->mnId < pKF->mnId)
                    continue;

                Eigen::Vector3f Ow2 = pCovKF->GetCameraCenter();
                glVertex3f(Ow.x(), Ow.y(), Ow.z());
                glVertex3f(Ow2.x(), Ow2.y(), Ow2.z());
            }
        }
        glEnd();
    }
}

void Viewer::DrawCurrentCamera(pangolin::OpenGlMatrix &M)
{
    const float w = mCameraSize;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    glPushMatrix();
    glMultMatrixd(M.m);

    glLineWidth(mCameraLineWidth);
    glColor3f(1.0f, 0.0f, 0.0f); // 当前相机：红色

    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glEnd();

    glPopMatrix();
}

void Viewer::SetCurrentCameraPose(const Eigen::Matrix4f &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexCamera);
    mCameraPose = Tcw;
}

void Viewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    Eigen::Matrix4f Twc;
    {
        std::unique_lock<std::mutex> lock(mMutexCamera);
        // mCameraPose 是 Tcw (World 到 Camera 的变换矩阵)
        // 求解其逆矩阵得到 Twc (Camera 到 World 的变换矩阵)
        Twc = mCameraPose.inverse();
    }

    // OpenGL 期待的是列主序 (Column-major) 矩阵格式
    M.m[0] = Twc(0, 0);
    M.m[4] = Twc(0, 1);
    M.m[8] = Twc(0, 2);
    M.m[12] = Twc(0, 3);
    M.m[1] = Twc(1, 0);
    M.m[5] = Twc(1, 1);
    M.m[9] = Twc(1, 2);
    M.m[13] = Twc(1, 3);
    M.m[2] = Twc(2, 0);
    M.m[6] = Twc(2, 1);
    M.m[10] = Twc(2, 2);
    M.m[14] = Twc(2, 3);
    M.m[3] = Twc(3, 0);
    M.m[7] = Twc(3, 1);
    M.m[11] = Twc(3, 2);
    M.m[15] = Twc(3, 3);
}

void Viewer::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Viewer::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
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