#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "Frame.h"
#include "Camera.h"
#include "Map.h"

class Initializer {
public:
    // 传入全局地图指针，初始化结果将直接写入地图
    Initializer();
    ~Initializer() = default;

    /**
     * @brief 尝试使用当前帧进行双目初始化
     * @param pFrame 当前输入的帧
     * @return true: 初始化成功; false: 视差不足或有效点太少，初始化失败
     */
    bool TryInitialize(std::shared_ptr<Frame> pFrame);

private:
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Camera> mpCamera;
};

#endif // INITIALIZER_H