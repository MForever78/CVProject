#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat harris(Mat &src, int block_size, int aperture_size, double k);
Mat heatmap(Mat &src);

int main(int argc, char **argv) {
    Mat src, src_gray;
    Size size = src.size();
    Mat dst(size, CV_32FC1);

    String standard_window_name = "Harris Corner Demo Standard";
    String window_name = "My Harris Corner";
    String heatmap_window_name = "Heatmap Result";

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
    namedWindow(heatmap_window_name, CV_WINDOW_AUTOSIZE);

    // Standard output
    cornerHarris( src_gray, dst, 3, 3, 0.04, BORDER_DEFAULT );
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs( dst_norm, dst_norm_scaled );
    imshow(standard_window_name, dst_norm_scaled);

    // My own output
    Mat result = harris(src_gray, 3, 3, 0.04);

    Mat result_norm, result_norm_scaled;
    
    // normalize to gray image
    normalize(result, result_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(result_norm, result_norm_scaled);
    //imwrite("gray_" + argv[1], result_norm_scaled);

    // show gray image
  	imshow(window_name, result_norm_scaled);
    
    //cout << result_norm_scaled << endl;

    // convert to heatmap image
    Mat heatimg = heatmap(result_norm_scaled);
    imshow(heatmap_window_name, heatimg);

  	waitKey(0);

    return 0;
}

Mat harris(Mat &src, int block_size, int aperture_size, double k) {
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
    
    boxFilter(cov, cov, cov.depth(), Size(block_size, block_size), Point(-1, -1), false);

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
        const unsigned char *data = src.ptr<unsigned char>(i);
        float *dst_data = dst.ptr<float>(i);

        for (int j = 0; j < size.width; ++j) {
            int idx1, idx2;
            float fraction;
            float gray_relative;

            gray_relative = data[j] / 255.0 * NUM_COLORS;
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

