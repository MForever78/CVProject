#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat harris(Mat &src, int aperture_size, double k);
Mat heatmap(Mat &src);
String filename;

int main(int argc, char **argv) {
    Mat src, src_gray;
    Size size = src.size();
    Mat dst(size, CV_32FC1);

    String window_name = "My Harris Corner";
    String heatmap_window_name = "Heatmap Result";
    String src_name = "Source Image Corner";

    // Load image
    filename = String(argv[1]);
    src = imread(argv[1]);
    if (!src.data) {
        printf("Cannot read image\n");
        exit(-1);
    }

    cvtColor(src, src_gray, CV_BGR2GRAY);
    
    // Create window
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(heatmap_window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(src_name, CV_WINDOW_AUTOSIZE);

    // My own output
    Mat result = harris(src_gray, 3, 0.04);

    Mat result_norm, result_norm_scaled;
    
    // normalize to gray image
    normalize(result, result_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(result_norm, result_norm_scaled);
    imshow(window_name, result_norm_scaled);
    
    int thresh = 50;

    Mat src_clone = src.clone();

    // show gray image
    for (int i = 0; i < result_norm.rows; ++i) {
        for (int j = 0; j < result_norm.cols; ++j) {
            if ((int) result_norm.at<float>(i, j) > thresh) {
                circle(src_clone, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    imshow(src_name, src_clone);

    imwrite("corner_" + filename, src_clone);
    
    // convert to heatmap image
    Mat result_heat_norm(size, CV_32FC1);
    normalize(result, result_heat_norm, 0, 1024767, NORM_MINMAX, CV_32FC1, Mat());
    Mat heatimg = heatmap(result_heat_norm);
    imshow(heatmap_window_name, heatimg);

    waitKey(0);

    return 0;
}

Mat harris(Mat &src, int aperture_size, double k) {
    // Generate Ix and Iy
    Mat dx, dy;

    int ddepth = CV_32F;

    // Gradient X
    Sobel(src, dx, ddepth, 1, 0, aperture_size);

    // Gradient Y
    Sobel(src, dy, ddepth, 0, 1, aperture_size);
    
    Size size = src.size();
    Mat dst(size, CV_32FC1);
    Mat cov(size, CV_32FC3);
    Mat cov_eigen_min(size, CV_32FC1);
    Mat cov_eigen_min_norm(size, CV_32FC1);
    Mat cov_eigen_max(size, CV_32FC1);
    Mat cov_eigen_max_norm(size, CV_32FC1);

    for (int i = 0; i < size.height; ++i) {
        float *cov_data = cov.ptr<float>(i);
        const float *dx_data = dx.ptr<float>(i);
        const float *dy_data = dy.ptr<float>(i);

        for (int j = 0; j < size.width; ++j) {
            float dx = dx_data[j];
            float dy = dy_data[j];
            
            cov_data[j*3] = dx * dx;
            cov_data[j*3+1] = dx * dy;
            cov_data[j*3+2] = dy * dy;
        }
    }

    boxFilter(cov, cov, cov.depth(), Size(3, 3), Point(-1, -1), false);
    
    for (int i = 0; i < size.height; ++i) {
        const float *cov_data = cov.ptr<float>(i);
        float *cov_eigen_min_data = cov_eigen_min.ptr<float>(i);
        float *cov_eigen_max_data = cov_eigen_max.ptr<float>(i);

        for (int j = 0; j < size.width; ++j) {
            Mat tensor(2, 2, CV_32FC1);
            Mat tensor_eigen(1, 2, CV_32FC1);
            tensor.at<float>(0, 0) = cov_data[j*3];
            tensor.at<float>(0, 1) = cov_data[j*3+1];
            tensor.at<float>(1, 0) = cov_data[j*3+1];
            tensor.at<float>(1, 1) = cov_data[j*3+2];
            
            eigen(tensor, tensor_eigen);
            cov_eigen_max_data[j] = tensor_eigen.at<float>(0);
            cov_eigen_min_data[j] = tensor_eigen.at<float>(1);
        }
    }

    normalize(cov_eigen_min, cov_eigen_min_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    imwrite("min_" + filename, cov_eigen_min_norm);
    normalize(cov_eigen_max, cov_eigen_max_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    imwrite("max_" + filename, cov_eigen_max_norm);

    for (int i = 0; i < size.height; ++i) {
        const float *min_data = cov_eigen_min.ptr<float>(i);
        const float *max_data = cov_eigen_max.ptr<float>(i);
        float *dst_data = dst.ptr<float>(i);

        for (int j = 0; j < size.width; ++j) {
            dst_data[j] = (float)(min_data[j] * max_data[j] - k * (min_data[j] + max_data[j]) * (min_data[j] + max_data[j]));
        }
    }

    return dst;
}

Mat heatmap(Mat &src) {
    // use 5 base colors
    const int NUM_COLORS = 5;
    static int color[NUM_COLORS][3] = {
        // blue
        {0, 0, 255},
        // cyan
        {0, 255, 255},
        // green
        {0, 255, 0},
        // yellow
        {255, 255, 0},
        // red
        {255, 0, 0}
    };

    Size size = src.size();

    Mat dst(size, CV_32FC3);

    for (int i = 0; i < size.height; ++i) {
        const float *data = src.ptr<float>(i);
        float *dst_data = dst.ptr<float>(i);

        for (int j = 0; j < size.width; ++j) {
            int idx1, idx2;
            float fraction;
            float gray_relative;

            gray_relative = data[j] / 1024768.0 * NUM_COLORS;
            idx1 = floor(gray_relative);
            idx2 = idx1 + 1;
            fraction = gray_relative - idx1;

            dst_data[j*3+2] = (float)(color[idx1][0] + (color[idx2][0] - color[idx1][0]) * fraction);
            dst_data[j*3+1] = (float)(color[idx1][1] + (color[idx2][1] - color[idx1][1]) * fraction);
            dst_data[j*3] = (float)(color[idx1][2] + (color[idx2][2] - color[idx1][2]) * fraction);
        }
    }

    return dst;
}

