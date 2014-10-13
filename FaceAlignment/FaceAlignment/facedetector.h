#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include "common.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "LibFace.h"
#include "Face.h"

using namespace libface;

class FaceDetector
{
public:
  FaceDetector();

  static void detectFace( const string &filename )
  {
    cout << "detecting face ..." << endl;

    /* read the input image */
    IplImage* image = cvLoadImage( filename.c_str(), 1 );

    /*
   create a b&w image of the same size object as a placeholder for now
   - cvSize(width, height) is the object representing dimensions of the image
   - 8 stands for 8-bit depth
   - 1 means one-channel (b&w)
   Hence a 24bit RGB image of size 800x600 would be created with
   cvCreateImage( cvSize(800, 600), 8, 3);
   */
    IplImage* gray = cvCreateImage( cvSize(image->width,image->height), 8, 1 );

    /* now convert the input image into b&w and store it in the placeholder */
    cvCvtColor( image, gray, CV_BGR2GRAY );

    libface::Mode mode = libface::DETECT;

    //Make an instance of LibFace class with appropriate parameters
    LibFace* libFace = new LibFace(mode);
    vector<Face*>* result = libFace->detectFaces(filename);

    cout << "detected faces: " << result->size() << endl;
    /* go through all the detected faces, and draw them into the input image */
    for (int i = 0; i < (result ? result->size() : 0); i++)
    {
      Face* face = (*result)[i];
      cout << face->getX1() << ", "
           << face->getY1() << ", "
           << face->getX2() << ", "
           << face->getY2() << endl;
      CvPoint ul; CvPoint lr;
      ul.x = face->getX1(); ul.y = face->getY1();
      lr.x = face->getX2(); lr.y = face->getY2();

      /* draws a rectangle with given coordinates of the upper left
    and lower right corners into an image */
      cvRectangle(image, ul, lr, Scalar(0, 0, 255), 3, 8, 0);
    }

    delete result;
    delete libFace;

    /* free up the memory */
    cvReleaseImage( &gray );

    return;

    /* create a window with handle result */
    cvNamedWindow( "result" );

    /* show the result and wait for a keystroke form user before finishing */
    cvShowImage( "result", image );
    cvWaitKey(0);
    cvReleaseImage( &image );
    cvDestroyWindow("result");
    return;
  }
};

#endif // FACEDETECTOR_H
