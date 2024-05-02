#include "thinning.hpp"

using namespace cv;

uint8_t markers2D[][3][3] = {
    //straight
    {
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0}
    },
    {
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0}
    },
    {
        {0, 0, 0},
        {1, 1, 1},
        {0, 0, 0}
    },
    {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    },
    //corner
    {
        {0, 1, 0},
        {0, 1, 1},
        {0, 0, 0}
    },
    {
        {0, 0, 0},
        {0, 1, 1},
        {0, 1, 0}
    },
    {
        {0, 0, 0},
        {1, 1, 0},
        {0, 1, 0}
    },
    {
        {0, 1, 0},
        {1, 1, 0},
        {0, 0, 0}
    },
    {
        {0, 1, 0},
        {0, 1, 0},
        {0, 0, 1}
    },
    {
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0}
    },
    {
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0}
    },
    {
        {0, 1, 0},
        {0, 1, 0},
        {1, 0, 0}
    },
    {
        {1, 0, 0},
        {0, 1, 1},
        {0, 0, 0}
    },
    {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 0}
    },
    {
        {0, 0, 1},
        {1, 1, 0},
        {0, 0, 0}
    },
    {
        {0, 0, 0},
        {1, 1, 0},
        {0, 0, 1}
    },
    //T
    {
        {0, 1, 0},
        {1, 1, 1},
        {0, 0, 0}
    },
    {
        {0, 0, 0},
        {1, 1, 1},
        {0, 1, 0}
    },
    {
        {0, 1, 0},
        {1, 1, 0},
        {0, 1, 0}
    },
    {
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0}
    },
    {
        {1, 0, 0},
        {0, 1, 0},
        {1, 0, 1}
    },
    {
        {1, 0, 1},
        {0, 1, 0},
        {0, 0, 1}
    },
    {
        {1, 0, 1},
        {0, 1, 0},
        {1, 0, 0}
    },
    {
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 1}
    },
    //cross
    {
        {0, 1, 0},
        {1, 1, 1},
        {0, 1, 0}
    },
    {
        {1, 0, 1},
        {0, 1, 0},
        {1, 0, 1}
    },


};

MapThinner2D::MapThinner2D()
{
}

MapThinner2D::~MapThinner2D()
{
}

void MapThinner2D::thinning(const cv::Mat &src, cv::Mat &cost, cv::Mat& thin)
{
    Mat src2 = src.clone();
    Mat prev;
    Mat diff;
    int iter = 0;
    Mat tempCost = Mat::zeros(src2.size(), CV_32FC1);
    while (true)
    {
        iter++;
        // tempCost = tempCost + Mat::ones(src2.size(), CV_32FC1);
        prev = Mat::zeros(src2.size(), CV_8UC1);
        for (int i = 1; i < src2.rows-1; i++)
        {
            for (int j = 1; j < src2.cols-1; j++)
            {
                if (src2.at<uchar>(i, j) == 0)
                {
                    continue;
                }
                uint8_t window[3][3] = {
                    {src2.at<uchar>(i-1, j-1), src2.at<uchar>(i-1, j), src2.at<uchar>(i-1, j+1)},
                    {src2.at<uchar>(i, j-1), src2.at<uchar>(i, j), src2.at<uchar>(i, j+1)},
                    {src2.at<uchar>(i+1, j-1), src2.at<uchar>(i+1, j), src2.at<uchar>(i+1, j+1)}
                };
                // if there is zero in the window
                if (window[0][0] == 0 || window[0][1] == 0 || window[0][2] == 0 ||
                    window[1][0] == 0 || window[1][2] == 0 ||
                    window[2][0] == 0 || window[2][1] == 0 || window[2][2] == 0)
                {
                    prev.at<uchar>(i, j) = 1;
                }
            }
        }
        
        src2 -= prev;
        Mat prevf;
        prev.convertTo(prevf, CV_32FC1);
        tempCost = tempCost + prevf * iter;

        float max = 0, min = std::numeric_limits<float>::max();
        for (int i = 0; i < src2.rows; i++)
        {
            for (int j = 0; j < src2.cols; j++)
            {
                if (tempCost.at<float>(i, j) > max)
                {
                    max = tempCost.at<float>(i, j);
                }
                if (tempCost.at<float>(i, j) < min)
                {
                    min = tempCost.at<float>(i, j);
                }
            }
        }
        Mat showf = (tempCost-min) / (max - min);

        Mat temp = Mat::zeros(src2.size(), CV_8UC1);
        for (int i = 1; i < showf.rows-1; i++)
        {
            for (int j = 1; j < showf.cols-1; j++)
            {
                float p1 = showf.at<float>(i, j);
                float p2 = showf.at<float>(i-1, j);
                float p3 = showf.at<float>(i-1, j+1);
                float p4 = showf.at<float>(i, j+1);
                float p5 = showf.at<float>(i+1, j+1);
                float p6 = showf.at<float>(i+1, j);
                float p7 = showf.at<float>(i+1, j-1);
                float p8 = showf.at<float>(i, j-1);
                float p9 = showf.at<float>(i-1, j-1);

                float d91 = p1 - p9;
                float d21 = p1 - p2;
                float d31 = p1 - p3;
                float d41 = p1 - p4;
                float d51 = p1 - p5;
                float d61 = p1 - p6;
                float d71 = p1 - p7;
                float d81 = p1 - p8;

                if((d91 >= 0 && d51 >= 0 && !(d91 == 0 && d51 == 0)) ||
                    (d21 >= 0 && d61 >= 0 && !(d21 == 0 && d61 == 0)) ||
                    (d31 >= 0 && d71 >= 0 && !(d31 == 0 && d71 == 0)) ||
                    (d41 >= 0 && d81 >= 0 && !(d41 == 0 && d81 == 0)))
                {
                    temp.at<uchar>(i, j) = 255;
                }
            }
        }

        // imshow("thinning", temp);
        // imshow("cost", showf);
        // waitKey(10);

        if (countNonZero(prev) == 0)
        {
            cost = showf.clone();
            thin = temp.clone();
            return;
        }
    }

}
