#include "detector.h"

cv::Point2f detect_circle(const cv::Mat& frame) {
    if (frame.empty()) {
        return cv::Point2f(-1, -1);
    }

    // 降低处理分辨率
    cv::Mat small_frame;
    float scale_factor = 0.5;
    cv::resize(frame, small_frame, cv::Size(), scale_factor, scale_factor);

    // 转换到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(small_frame, hsv, cv::COLOR_BGR2HSV);

    // 定义亮橙色的HSV范围
    cv::Scalar lower_orange1(172, 110, 95);
    cv::Scalar upper_orange1(180, 255, 255);
    cv::Scalar lower_orange2(0, 110, 95);
    cv::Scalar upper_orange2(10, 255, 255);

    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lower_orange1, upper_orange1, mask1);
    cv::inRange(hsv, lower_orange2, upper_orange2, mask2);
    cv::bitwise_or(mask1, mask2, mask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return cv::Point2f(-1, -1);
    }

    double max_area = 0;
    int max_contour_idx = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_contour_idx = i;
        }
    }

    if (max_contour_idx == -1) {
        return cv::Point2f(-1, -1);
    }
    
    // 找到最小外接圆
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contours[max_contour_idx], center, radius);
    if (radius < 5) {
        return cv::Point2f(-1, -1);
    }

    center.x /= scale_factor;
    center.y /= scale_factor;

    return center;
}