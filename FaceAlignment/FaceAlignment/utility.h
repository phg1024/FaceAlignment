#pragma once

#include "opencv2/highgui/highgui.hpp"
using namespace cv;

template <typename T>
void showPointsWithImage(const cv::Mat &img, const T& pts1, const T& pts2) {
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
		circle(outimg, Point(x, y), 2, Scalar(0, 255, 255));
	}

	imshow("image", outimg);
	waitKey(0);
	destroyWindow("image");
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
	waitKey(0);
	destroyWindow("image");
}