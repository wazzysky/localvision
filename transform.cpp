#include "transform.h"
#include <iostream>

Transformer::Transformer() : initialized_(false) {}

bool Transformer::load_matrix(const std::string& filename) {
    try {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "ERROR: Could not open homography file: " << filename << std::endl;
            return false;
        }
        fs["homography_matrix"] >> homography_matrix_;
        fs.release();
        if (homography_matrix_.empty()) {
            std::cerr << "ERROR: Failed to read homography matrix from " << filename << std::endl;
            return false;
        }
        initialized_ = true;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception while reading file: " << e.what() << std::endl;
        return false;
    }
}

cv::Point2f Transformer::camera_to_world(const cv::Point2f& camera_pt) const {
    if (!initialized_) {
        return cv::Point2f(NAN, NAN);
    }

    std::vector<cv::Point2f> src_pts = {camera_pt};
    std::vector<cv::Point2f> dst_pts;

    cv::perspectiveTransform(src_pts, dst_pts, homography_matrix_);
    
    return dst_pts[0];
}