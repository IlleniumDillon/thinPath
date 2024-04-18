#include <opencv2/opencv.hpp>

void thinningIteration(cv::Mat& img, int iter)
{
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    for (int i = 1; i < img.rows-1; i++)
    {
        for (int j = 1; j < img.cols-1; j++)
        {
            uchar p2 = img.at<uchar>(i-1, j);
            uchar p3 = img.at<uchar>(i-1, j+1);
            uchar p4 = img.at<uchar>(i, j+1);
            uchar p5 = img.at<uchar>(i+1, j+1);
            uchar p6 = img.at<uchar>(i+1, j);
            uchar p7 = img.at<uchar>(i+1, j-1);
            uchar p8 = img.at<uchar>(i, j-1);
            uchar p9 = img.at<uchar>(i-1, j-1);

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                     (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) & m == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    img &= ~marker;
}

void thinning(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();
    dst /= 255;         // convert to binary image

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}
constexpr int mapWidth = 21;
constexpr int mapHeight = 16;
int main()
{
    uchar data[mapHeight][mapWidth] = {
        #include "MapAIUS3011"
    };
    cv::Mat src(mapHeight, mapWidth, CV_8UC1, data);
    src *= 255;
    src = 255 - src; 
    cv::resize(src, src, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
    cv::Mat bw;
    bw = src.clone();  
    thinning(bw, bw);

    bw = (bw + src) / 2;

    cv::imshow("src", src);
    cv::imshow("dst", bw);
    cv::waitKey(0);

    return 0;
}