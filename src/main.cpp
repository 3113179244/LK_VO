#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

int main()
{
    std::string file_path = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/00/";
    std::string left_dir = file_path + "image_0/";
    std::string right_dir = file_path + "image_1/";
    std::string times_path = file_path + "times.txt";

    if (!cv::utils::fs::exists(left_dir) || !cv::utils::fs::exists(right_dir))
    {
        std::cerr << "错误: 找不到 image_0 或 image_1 路径！" << std::endl;
        return -1;
    }

    std::vector<double> timestamps;
    std::ifstream times_file(times_path);
    if (!times_file.is_open())
    {
        std::cerr << "错误: 无法打开时间戳文件 " << times_path << std::endl;
        return -1;
    }

    double timestamp;
    while (times_file >> timestamp)
    {
        timestamps.push_back(timestamp);
    }
    times_file.close();

    std::cout << "成功加载 " << timestamps.size() << " 个时间戳。" << std::endl;
    std::cout << "  - 空格键 (Space): 暂停播放" << std::endl;
    std::cout << "  - Q 键           : 继续播放" << std::endl;
    std::cout << "  - ESC 键         : 退出程序" << std::endl;

    int frame_id = 0;
    bool is_paused = false;

    while (true)
    {
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << frame_id << ".png";
        std::string filename = ss.str();

        std::string left_img_path = left_dir + filename;
        std::string right_img_path = right_dir + filename;

        if (!cv::utils::fs::exists(left_img_path) || !cv::utils::fs::exists(right_img_path) || frame_id >= timestamps.size())
        {
            std::cout << "\n已到达序列末尾，播放结束。共处理 " << frame_id << " 帧。" << std::endl;
            break;
        }

        if (!is_paused)
        {
            cv::Mat img_left = cv::imread(left_img_path, cv::IMREAD_GRAYSCALE);
            cv::Mat img_right = cv::imread(right_img_path, cv::IMREAD_GRAYSCALE);

            if (img_left.empty() || img_right.empty())
            {
                std::cerr << "错误: 无法读取图像: " << filename << std::endl;
                break;
            }

            double current_timestamp = timestamps[frame_id];

            std::stringstream text_ss;
            text_ss << "Frame: " << frame_id << " | Time: " << std::fixed << std::setprecision(4) << current_timestamp << "s";
            
            cv::putText(img_left, text_ss.str(), cv::Point(30, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);

            std::string status_text = is_paused ? "PAUSED" : "PLAYING";
            cv::putText(img_left, status_text, cv::Point(img_left.cols - 160, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);

            cv::imshow("KITTI Mono Reader (Left)", img_left);
        }

        int wait_time = is_paused ? 0 : 20;
        char key = static_cast<char>(cv::waitKey(wait_time));

        if (key == 27)
        {
            std::cout << "\n按下 ESC，退出程序。" << std::endl;
            break;
        }
        else if (key == ' ')
        { 
            if (!is_paused)
            {
                is_paused = true;
                std::cout << "\r[状态] 已暂停播放 (按 Q 键继续)... " << std::flush;
            }
        }
        else if (key == 'q' || key == 'Q')
        { 
            if (is_paused)
            {
                is_paused = false;
                std::cout << "\r[状态] 恢复播放...               " << std::flush;
            }
        }

        if (!is_paused)
        {
            frame_id++;
        }
    }

    cv::destroyAllWindows();
    return 0;
}