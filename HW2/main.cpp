#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat harris(Mat &src, int block_size, int aperture_size, double k);

int main(int argc, char **argv) {
    Mat src, src_gray;
    Size size = src.size();
    Mat dst(size, CV_32FC1);

    String standard_window_name = "Harris Corner Demo Standard";
    String window_name = "My Harris Corner";

    // Load image
    src = imread(argv[1]);
    if (!src.data) {
        printf("Cannot read image\n");
        exit(-1);
    }

    cvtColor(src, src_gray, CV_BGR2GRAY);
    
    // Create window
    namedWindow(standard_window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);

    // Standard output
    cornerHarris( src_gray, dst, 3, 3, 0.04, BORDER_DEFAULT );
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs( dst_norm, dst_norm_scaled );
    imshow(standard_window_name, dst_norm_scaled);

    // My own output
    Mat result = harris(src, 3, 3, 0.04);

    Mat result_norm, result_norm_scaled;
    
    normalize(result, result_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(result_norm, result_norm_scaled);

  	imshow(window_name, result_norm_scaled);
  	waitKey(0);

    return 0;
}

Mat harris(Mat &src, int block_size, int aperture_size, double k) {
    // Generate Ix and Iy
    Mat dx, dy;

    int ddepth = CV_16S;

    // Gradient X
    Sobel(src, dx, ddepth, 1, 0, aperture_size);

    // Gradient Y
    Sobel(src, dy, ddepth, 0, 1, aperture_size);
    
    Size size = src.size();
    Mat dst(size, CV_32FC1);
    Mat cov(size, CV_32FC3);

    // Concate M matrix
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

    size.width *= size.height;
    size.height = 1;

    for (int i = 0; i < size.height; ++i) {
        const float *cov_data = cov.ptr<float>(i);
        float *dst_data = dst.ptr<float>(i);

        for (int j = 0; j < size.width; ++j) {
            float a = cov_data[j*3];
            float b = cov_data[j*3+1];
            float c = cov_data[j*3+2];

            dst_data[j] = (float)(a*c - b*b - k*(a + c)*(a + c));
        }
    }

    return dst;
}

