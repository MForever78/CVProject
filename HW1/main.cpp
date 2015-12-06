#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <boost/filesystem.hpp>

using namespace cv;

Mat clipHeight(Mat &orig, int maxHeight) {
    if (orig.cols <= maxHeight) return orig;
    return orig(Rect(0, (orig.rows - maxHeight) / 2, orig.cols, maxHeight));
}

std::vector<Mat> clipImages(std::vector<Mat> &images, int width, int height) {
    std::vector<Mat> ret;
    for (auto &it: images) {
        Mat standard = Mat::zeros(height, width, CV_8UC3);
        if (it.cols > width) {
            // needs clip
            double ratio = (double)it.cols / width;
            // zoom the height accordingly
            int newHeight = it.rows * ratio;
            // if new height still larger than standard height, clip it
            newHeight = newHeight > height ? height : newHeight;
            resize(it, it, Size(width, newHeight));
            int baseHeight = (height - newHeight) / 2;

            Mat crop = clipHeight(it, height);
            crop.copyTo(standard(Rect(0, baseHeight, width, newHeight)));
        } else {
            int newHeight = it.rows > height ? height : it.rows;
            Mat crop = clipHeight(it, height);

            int baseHeight = (height - newHeight) / 2;
            int baseWidth = (width - it.cols) / 2;

            crop.copyTo(standard(Rect(baseWidth, baseHeight, it.cols, newHeight)));
        }
        ret.push_back(standard);
    }

    return ret;
}

void appendVideo(VideoWriter &dist, VideoCapture &src) {
    Mat srcFrame;
    for (;;) {
        src >> srcFrame;
        if (srcFrame.empty()) break;
        dist << srcFrame;
    }
}

void darken(Mat &image, double percent) {
    int rows = image.rows;
    int cols = image.cols * 3;
    uchar *p;

    for (int i = 0; i < rows; ++i) {
        p = image.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            p[j] -= p[j] * percent / 100;
        }
    }
}

Mat transit(Mat &image, double percent) {
    int rows = image.rows;
    int cols = image.cols * 3;
    uchar *p, *pi;
    Mat ret(image.size(), image.type());

    for (int i = 0; i < rows; ++i) {
        p = ret.ptr<uchar>(i);
        pi = image.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            p[j] = pi[j] * percent / 100;
        }
    }

    return ret;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: Display <path to media directory>" << std::endl;
        exit(-1);
    }

    namespace fs = boost::filesystem;

    fs::path mediaPath(argv[1]);
    
    try {
        if (fs::exists(mediaPath) && fs::is_directory(mediaPath)) {
            std::vector<Mat> images;
            VideoCapture inputVideo;

            // get iamges and video
            for (fs::directory_entry &x : fs::directory_iterator(mediaPath)) {
                if (x.path().extension().compare(".jpg") == 0) {
                    images.push_back(imread(x.path().string(), 1));
                } else if (x.path().extension().compare(".avi") == 0) {
                    inputVideo = VideoCapture(x.path().string());
                }
            }

            int width = (int)inputVideo.get(CV_CAP_PROP_FRAME_WIDTH);
            int height = (int)inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT);

            std::vector<Mat> clippedImages = clipImages(images, width, height);
            
            std::cout << width << "x" << height << std::endl;

            double fps = 24;

            VideoWriter outVideo = VideoWriter("out.avi", VideoWriter::fourcc('X', '2', '6', '4'), fps, Size(width, height));

            for (auto &it: clippedImages) {
                for (int i = 0; i < 1 * fps; ++i) {
                    Mat transition = transit(it, 100.0 / fps * i);
                    outVideo << transition;
                }
                for (int i = 0; i < 2 * fps; ++i) {
                    outVideo << it;
                }
                for (int i = 0; i < 1 * fps; ++i) {
                    darken(it, 100.0 / fps * i);
                    outVideo << it;
                }
            }

            appendVideo(outVideo, inputVideo);
        } else {
            std::cout << "Not a directory!" << std::endl;
        }
    }

    catch (const fs::filesystem_error &ex) {
        std::cout << ex.what() << std::endl;
    }

    return 0;
}

