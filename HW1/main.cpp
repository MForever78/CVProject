#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: display <path to image>\n");
        exit(-1);
    }

    Mat image;
    image = imread(argv[1], 1);

    if (!image.data) {
        printf("Not a image!\n");
        exit(-1);
    }

    namedWindow("OpenCV Demo", WINDOW_AUTOSIZE);
    imshow("OpenCN Demo", image);

    waitKey(0);

    return 0;
}

