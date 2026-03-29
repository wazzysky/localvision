#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class Transformer {
    public:
        Transformer();
        bool load_matrix(const std::string& filename);
        cv::Point2f camera_to_world(const cv::Point2f& camera_pt) const;

    private:
        cv::Mat homography_matrix_;
        bool initialized_ = false;
};