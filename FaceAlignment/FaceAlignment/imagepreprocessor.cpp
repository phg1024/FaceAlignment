#include "imagepreprocessor.h"

#include "Utils/stringutils.h"

#include "opencv2/highgui/highgui.hpp"
using namespace cv;

ImagePreprocessor::ImagePreprocessor(const string &imgfile, const string &ptsfile) {

  process(imgfile, ptsfile);

  //waitKey(0); //wait infinite time for a keypress

  //exit(0);
  //std::this_thread::sleep_for (std::chrono::seconds(1));
}

void ImagePreprocessor::process(const string &imgfile, const string &ptsfile) {
  Mat img = imread(imgfile.c_str(), CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'

  if (img.empty()) //check whether the image is loaded or not
  {
       cout << "Error : Image cannot be loaded..!!" << endl;
       return;
  }

  /// read in the points
  ifstream f(ptsfile);
  string version_tag;
  f >> version_tag;
  std::getline(f, version_tag);
  string pointcount_tag;
  int npoints;
  f >> pointcount_tag >> npoints;
  cout << pointcount_tag << endl;
  cout << npoints << " points" << endl;
  f.ignore();
  string dummy;
  std::getline(f, dummy);

  struct point_t {
    double x, y;
  };
  vector<point_t> pts(npoints);

  for(int i=0;i<npoints;++i) {
    f >> pts[i].x >> pts[i].y;
    cout << pts[i].x << ',' << pts[i].y << endl;

    circle(img, Point(pts[i].x, pts[i].y), 2, Scalar(0, 255, 0));
  }

  namedWindow(imgfile.c_str(), CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
  imshow(imgfile.c_str(), img); //display the image which is stored in the 'img' in the "MyWindow" window

  f.close();
}
