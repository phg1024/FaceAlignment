#include "imagepreprocessor.h"
#include "facedetector.h"

#include "Utils/stringutils.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

ImagePreprocessor::ImagePreprocessor() {}

void ImagePreprocessor::process(const string &imgfile, const string &ptsfile) {
  FaceDetector::detectFace(imgfile);


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
    point_t(){}
    point_t(double x, double y):x(x), y(y){}
    double x, y;
  };
  vector<point_t> pts(npoints);

  point_t maxpt(0.0, 0.0), minpt(img.cols, img.rows);

  for(int i=0;i<npoints;++i) {
    double x, y;
    f >> x >> y;
    cout << x << ',' << y << endl;
    pts[i].x = x; pts[i].y = y;

    maxpt.x = max(x, maxpt.x); maxpt.y = max(y, maxpt.y);
    minpt.x = min(x, minpt.x); minpt.y = min(y, minpt.y);

    //circle(img, Point(pts[i].x, pts[i].y), 2, Scalar(0, 255, 0));
  }
  f.close();

  //rectangle(img, Point(minpt.x, minpt.y), Point(maxpt.x, maxpt.y), Scalar(0, 0, 255));

  point_t center(0.5*(minpt.x+maxpt.x), 0.5*(minpt.y+maxpt.y));
  double maxdim = max(img.rows, img.cols);
  maxdim = min(maxdim, img.rows-1-center.y);
  maxdim = min(maxdim, center.y);
  maxdim = min(maxdim, img.cols-1-center.x);
  maxdim = min(maxdim, center.x);
  maxdim = (int)maxdim;
  cout << "maxdim = " << maxdim * 2 << endl;

  double w = maxpt.x - minpt.x, h = maxpt.y - minpt.y;
  double scale = 4.0;
  w *= scale;
  h *= scale;
  double imgsize = min(max(w, h), maxdim*2);
  minpt.x = center.x - 0.5 * imgsize; minpt.y = center.y - 0.5 * imgsize;
  maxpt.x = center.x + 0.5 * imgsize; maxpt.y = center.y + 0.5 * imgsize;
  //rectangle(img, Point(minpt.x, minpt.y), Point(maxpt.x, maxpt.y), Scalar(0, 0, 255));

  //namedWindow(imgfile.c_str(), CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
  //imshow(imgfile.c_str(), img); //display the image which is stored in the 'img' in the "MyWindow" window

  // take a submatrix
  Mat subimg = img(cv::Rect(minpt.x, minpt.y, imgsize, imgsize));

  // scale it
  const double targetSize = 256.0;
  Size size(targetSize, targetSize);
  Mat regimg; // regular image
  resize(subimg, regimg, size);//resize image

  // convert to grayscale image
  Mat outimg;
  if( regimg.channels() == 3 ) {
    cvtColor( regimg, outimg, CV_BGR2GRAY );
  }
  else {
    outimg = regimg;
  }

  // scaling factor
  double sfactor = targetSize / imgsize;
  cout << "scale = " << sfactor << endl;

  // transform all points
  vector<point_t> npts(npoints);
  for(int i=0;i<npoints;++i) {
    point_t npi = pts[i];
    npi.x = (npi.x - minpt.x) * sfactor;
    npi.y = (npi.y - minpt.y) * sfactor;
    npts[i] = npi;
    //circle(outimg, Point(npi.x, npi.y), 2, Scalar(0, 255, 0));
  }

  //namedWindow("cutted", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
  //imshow("cutted", outimg); //display the image which is stored in the 'img' in the "MyWindow" window

  // save it
  string outputfile = imgfile.substr(0, imgfile.size()-4) + "_cutted.png";
  imwrite(outputfile, outimg);

  // save a version with points
  for(int i=0;i<npoints;++i) {
    point_t npi = npts[i];
    circle(regimg, Point(npi.x, npi.y), 2, Scalar(0, 255, 0));
  }
  string outputfile2 = imgfile.substr(0, imgfile.size()-4) + "_cutted_with_points.png";
  imwrite(outputfile2, regimg);

  string outptsfile = ptsfile.substr(0, ptsfile.size()-4) + "_cutted.pts";
  ofstream fout(outptsfile);
  fout << npoints << endl;
  for(int i=0;i<npoints;++i) {
    fout << npts[i].x << ' ' << npts[i].y << endl;
  }
  fout.close();


  //waitKey(0); //wait infinite time for a keypress

  //exit(0);
  //std::this_thread::sleep_for (std::chrono::seconds(1));
}
