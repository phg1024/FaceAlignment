#ifndef TESTTRANSFORM_H
#define TESTTRANSFORM_H

#include "transform.hpp"

namespace FATest {
namespace TransformTest {

void testTransform() {
  cout << "testing tranformation utilities ..." << endl;
  // Seed with a real random value, if available
  std::random_device rd;

  // Choose a random mean between 1 and 6
  std::default_random_engine e1(rd());
  std::uniform_int_distribution<int> uniform_dist(0, 10);

  const int n = 5;
  vec pts(n*2);
  for(int i=0;i<n;++i) {
    pts(i*2) = uniform_dist(e1);
    pts(i*2+1) = uniform_dist(e1);
  }
  Matrix3x3<double> M(0.4330, -0.2500, 0.2500,
                      0.2500,  0.4330, 0.1000,
                      0, 0, 1.0);

  vec newpts = pts;
  for(int i=0;i<n;++i) {
    Point2<double> npt = M * Point2<double>(pts(i*2), pts(i*2+1));
    newpts(i*2) = npt.x + uniform_dist(e1) * 0.01;
    newpts(i*2+1) = npt.y + uniform_dist(e1) * 0.01;
  };

  cout << "estimating transformation matrix ..." << endl;
  Matrix3x3<double> Mest = Transform<double>::estimateTransformMatrix(pts, newpts);
  Matrix3x3<double> Mest_cv = Transform<double>::estimateTransformMatrix_cv(pts, newpts);
  cout << M << endl;
  cout << Mest << endl;
  cout << Mest_cv << endl;

  cout << "done." << endl;
}
}
}

#endif // TESTTRANSFORM_H
