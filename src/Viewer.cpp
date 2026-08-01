#include "Viewer.h"
#include <chrono>
#include <thread>

Viewer::Viewer(System* pSystem, std::shared_ptr<Map> pMap)
    : mpSystem(pSystem), mpMap(pMap), mbStopRequested(false)
{
}

void Viewer::Run()
{
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(mMutexStop);
            if (mbStopRequested)
                break;
        }

        // 渲染与休眠逻辑（后续可在此处接入 Pangolin 绘图）
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

void Viewer::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
}