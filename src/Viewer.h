#ifndef VIEWER_H
#define VIEWER_H

#include <memory>
#include <mutex>

class System;
class Map;

class Viewer
{
public:
    Viewer(System* pSystem, std::shared_ptr<Map> pMap);
    ~Viewer() = default;

    // 线程主运行函数
    void Run();

    // 停止请求接口
    void RequestStop();

private:
    System* mpSystem;
    std::shared_ptr<Map> mpMap;

    bool mbStopRequested;
    std::mutex mMutexStop;
};

#endif // VIEWER_H