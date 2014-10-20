#pragma once

#include "opencv2/highgui/highgui.hpp"
using namespace cv;

#include <chrono>
#include <thread>

void waitFor(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

inline void showImage(const cv::Mat &img, const string &title="image") {
  namedWindow(title, CV_WINDOW_AUTOSIZE);
  imshow(title, img);
  waitKey(0);
  destroyWindow(title);
}

template <typename T>
void showPointsWithImage(const string &title, const cv::Mat &img, const T& pts) {
  cv::Mat outimg;
  cvtColor(img, outimg, CV_GRAY2RGB);
  namedWindow(title, CV_WINDOW_AUTOSIZE);

  int npts = pts.n_elem / 2;
  for (int i = 0; i<npts; ++i) {
    double x = pts(i * 2);
    double y = pts(i * 2 + 1);
    circle(outimg, Point(x, y), 2, Scalar(0, 255, 0));
  }

  imshow(title, outimg);
#if 1
  waitKey(0);
#else
  waitFor(500);
#endif
  destroyWindow(title);
}

template <typename T>
void showPointsWithImage(const string &title, const cv::Mat &img, const T& pts1, const T& pts2) {
	cv::Mat outimg;
	cvtColor(img, outimg, CV_GRAY2RGB);
  namedWindow(title, CV_WINDOW_AUTOSIZE);

	int npts = pts1.n_elem / 2;
	for (int i = 0; i<npts; ++i) {
		double x = pts1(i * 2);
		double y = pts1(i * 2 + 1);
		circle(outimg, Point(x, y), 2, Scalar(0, 255, 0));

		x = pts2(i * 2);
		y = pts2(i * 2 + 1);
		circle(outimg, Point(x, y), 2, Scalar(0, 255, 255));
	}

  imshow(title, outimg);
#if 1
  waitKey(0);
#else
  waitFor(500);
#endif
  destroyWindow(title);
}

template <typename T>
void showPointsWithImage(const cv::Mat &img, const T& pts1, const T& pts2, const T& pts3) {
	cv::Mat outimg;
	cvtColor(img, outimg, CV_GRAY2RGB);
	namedWindow("image", CV_WINDOW_AUTOSIZE);

	int npts = pts1.n_elem / 2;
	for (int i = 0; i<npts; ++i) {
		double x = pts1(i * 2);
		double y = pts1(i * 2 + 1);
		circle(outimg, Point(x, y), 2, Scalar(0, 255, 0));

		x = pts2(i * 2);
		y = pts2(i * 2 + 1);
		circle(outimg, Point(x, y), 2, Scalar(0, 0, 255));

		x = pts3(i * 2);
		y = pts3(i * 2 + 1);
		circle(outimg, Point(x, y), 2, Scalar(0, 255, 255));
	}

	imshow("image", outimg);
#if 1
	waitKey(0);
#else
  waitFor(500);
#endif
  destroyWindow("image");
}
