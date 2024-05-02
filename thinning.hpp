#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

class MapThinner2D
{
public:
    MapThinner2D();
    ~MapThinner2D();

    void thinning(const cv::Mat &src, cv::Mat &cost, cv::Mat& thin);
private:
    std::vector<cv::Mat> markers;
};