#ifndef TESTTRANSFORM_H
#define TESTTRANSFORM_H

#include "transform.hpp"

namespace FATest {
namespace TransformTest {

void testTransform() {
  // Seed with a real random value, if available
  std::random_device rd;

  // Choose a random mean between 1 and 6
  std::default_random_engine e1(rd());
  std::uniform_int_distribution<int> uniform_dist(0, 10);

  const int n = 5;
  vector<Point2<double>> pts;
  for(int i=0;i<n;++i) {
    pts.push_back(Point2<double>(uniform_dist(e1), uniform_dist(e1)));
  }
  Matrix3x3<double> M(0.4330, -0.2500, 0.2500,
                      0.2500,  0.4330, 0.1000,
                      0, 0, 1.0);

  vector<Point2<double>> newpts = pts;
  for_each(newpts.begin(), newpts.end(), [&](Point2<double> &pt){
    Point3<double> npt = M * Point3<double>(pt.x, pt.y, 1.0);
    pt.x = npt.x + uniform_dist(e1) * 0.05; pt.y = npt.y + uniform_dist(e1) * 0.05;
  });

  Matrix3x3<double> Mest = Transform<double>::estimateTransformMatrix(pts, newpts);
  cout << M << endl;
  cout << Mest << endl;
}
}
}

#endif // TESTTRANSFORM_H
