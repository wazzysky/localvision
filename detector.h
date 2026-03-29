#pragma once

#include <opencv2/opencv.hpp>

// 检测图像中的圆形目标
// @param frame 输入的BGR图像
// @return 检测到的圆心坐标，如果未检测到则返回cv::Point(-1, -1)
cv::Point2f detect_circle(const cv::Mat& frame);