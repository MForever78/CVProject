#include <cstdio>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: display <path to image>\n");
        exit(-1);
    }

    Mat image;
    // load with 3 channels 
    image = imread(argv[1], 1);

    if (!image.data) {
        printf("Not a image!\n");
        exit(-1);
    }

    int nRows = image.rows;
    int nCols = image.cols;
    int cx = nCols / 2;
    int cy = nRows / 2;
    double scale = -1;
    uchar *p;

    for (int i = 0; i < nRows; i++) {
        p = image.ptr<uchar>(i);
        for (int j = 0; j < nCols * 3; j += 3) {
            double dx=(double)(j/3-cx)/cx;
            double dy=(double)(i-cy)/cy;
            double weight=exp((dx*dx+dy*dy)*scale);
            p[j] *= weight;
            p[j+1] *= weight;
            p[j+2] *= weight;
        }
    }
    imshow("OpenCV Demo", image);

    waitKey(0);

    return 0;
}

