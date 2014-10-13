#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Geometry/matrix.hpp"
#include "Utils/utility.hpp"
using namespace PhGUtils;

#include <armadillo>
using namespace arma;

template <typename T>
class Transform
{
public:
  static Matrix3x3<T> estimateTransformMatrix(const vector<Point2<T>> &from,
                                              const vector<Point2<T>> &to);
  static vector<Point2<T>> transform(const vector<Point2<T>> &pts,
                                     const Matrix3x3<T> &M);

};

template <typename T>
vector<Point2<T>> Transform<T>::transform(const vector<Point2<T> > &pts, const Matrix3x3<T> &M)
{
  vector<Point2<T>> pts_trans = pts;
  for(auto &p : pts_trans) {
    p = M * p;
  }
  return pts_trans;
}

template <typename T>
Matrix3x3<T> Transform<T>::estimateTransformMatrix(const vector<Point2<T> > &p, const vector<Point2<T>> &q)
{
  //  % MATLAB implementation
  //  function [s, R, t] = estimateTransform(p, q)
  //  [n, m] = size(p);

  //  mu_p = mean(p);
  //  mu_q = mean(q);

  //  dp = p - repmat(mu_p, n, 1);
  //  sig_p2 = sum(sum(dp .* dp))/n;

  //  dq = q - repmat(mu_q, n, 1);
  //  sig_q2 = sum(sum(dq .* dq))/n;

  //  sig_pq = dq' * dp / n;

  //  det_sig_pq = det(sig_pq);
  //  S = diag(ones(m, 1));
  //  if det_sig_pq < 0
  //      S(n, m) = -1;
  //  end

  //  [U, D, V] = svd(sig_pq);

  //  R = U * S * V';
  //  s = trace(D*S)/sig_p2;
  //  t = mu_q' - s * R * mu_p';
  //  end

  //cout << "estimating tranformation matrix ..." << endl;

  assert(p.size() == q.size());
  int n = p.size();
  assert(n>0);
  const int m = 2;

  cout << "n = " << n << endl;

  mat pmat(n, 2), qmat(n, 2);
  for(int i=0;i<n;++i) {
    pmat(i, 0) = p[i].x;
    pmat(i, 1) = p[i].y;

    qmat(i, 0) = q[i].x;
    qmat(i, 1) = q[i].y;
  }

  mat mu_p = mean(pmat);
  mat mu_q = mean(qmat);

  mat dp = pmat - repmat(mu_p, n, 1);
  mat dq = qmat - repmat(mu_q, n, 1);

  double sig_p2 = sum(sum(dp % dp))/n;
  double sig_q2 = sum(sum(dq % dq))/n;

  mat sig_pq = trans(dq) * dp / n;

  double det_sig_pq = det(sig_pq);
  mat S = eye(m, m);
  if( det_sig_pq < 0 ) S(m, m) = -1;

  mat U, V;
  vec D;
  svd(U, D, V, sig_pq);

  cout << U << endl;
  cout << D << endl;
  cout << V << endl;

  mat R = U * S * trans(V);
  cout << R << endl;
  double s = trace(diagmat(D) * S)/sig_p2;
  vec t = trans(mu_q) - s * R * trans(mu_p);

  return Matrix3x3<T>(s * R(0, 0), s * R(0, 1), t(0),
           s * R(1, 0), s * R(1, 1), t(1),
           0, 0, 1);
}

#endif // TRANSFORM_H
