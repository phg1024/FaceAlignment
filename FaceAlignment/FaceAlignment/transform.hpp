#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Geometry/matrix.hpp"
#include "Utils/utility.hpp"
using namespace PhGUtils;

#include "numerical.hpp"

#include "opencv2/video/video.hpp"


template <typename T>
class Transform
{
public:
  static pair<Matrix2x2<T>, Point2<T>> estimateTransformMatrix(const arma::vec &from,
                                              const arma::vec &to);
  static vector<Point2<T>> transform(const vector<Point2<T>> &pts,
                                     const Matrix2x2<T> &M);

  static pair<Matrix2x2<T>, Point2<T>> estimateTransformMatrix_cv(const arma::vec &from,
                                           const arma::vec &to);

  static arma::vec transform(const arma::vec &v,
                             const Matrix2x2<T> &M);

};

template <typename T>
vector<Point2<T>> Transform<T>::transform(const vector<Point2<T> > &pts, const Matrix2x2<T> &M)
{
  vector<Point2<T>> pts_trans = pts;
  for(auto &p : pts_trans) {
    p = M * p;
  }
  return pts_trans;
}

template <typename T>
arma::vec Transform<T>::transform(const arma::vec &v, const Matrix2x2<T> &M)
{
  int npts = v.n_elem / 2;
  arma::vec res(v.n_elem);
  for(int i=0, j=0;i<npts;++i, j+=2) {
    double x = v(j), y = v(j+1);
    auto npt = M * Point2<T>(x, y);
    res(j) = npt.x;
    res(j+1) = npt.y;
  }
  return res;
}


template <typename T>
pair<Matrix2x2<T>, Point2<T>> Transform<T>::estimateTransformMatrix_cv(const arma::vec &p, const arma::vec &q)
{
  int n = p.n_elem / 2;
  vector<cv::Point2f> src(n), dst(n);
  for(int i=0;i<n;++i) {
    src[i].x = p(i*2);
    src[i].y = p(i*2+1);
    dst[i].x = q(i*2);
    dst[i].y = q(i*2+1);
  }
  cv::Mat M = cv::estimateRigidTransform(src, dst, false);
  if (M.empty()) {
    //cout << "empty matrix" << endl;
	  return Transform<T>::estimateTransformMatrix(p, q);
  }
  else
	  return make_pair(
		Matrix2x2<T>(M.at<double>(0, 0), M.at<double>(0, 1),
		M.at<double>(1, 0), M.at<double>(1, 1)),
		Point2<T>(M.at<double>(0, 2), M.at<double>(1, 2)));
}

template <typename T>
pair<Matrix2x2<T>, Point2<T>> Transform<T>::estimateTransformMatrix(const arma::vec &p, const arma::vec &q)
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

  //cout << p.n_elem << ", " << q.n_elem << endl;
  assert(p.n_elem == q.n_elem);

  int n = p.n_elem / 2;
  assert(n>0);
  const int m = 2;

  //cout << "n = " << n << endl;

  mat pmat = p, qmat = q;
  pmat.reshape(n, 2);
  qmat.reshape(n, 2);

  mat mu_p = mean(pmat);
  mat mu_q = mean(qmat);

  mat dp = pmat - repmat(mu_p, n, 1);
  mat dq = qmat - repmat(mu_q, n, 1);

  double sig_p2 = sum(sum(dp % dp))/n;
  double sig_q2 = sum(sum(dq % dq))/n;

  mat sig_pq = trans(dq) * dp / n;

  double det_sig_pq = det(sig_pq);
  mat S = eye(m, m);
  //if( det_sig_pq < 0 ) S(m-1, m-1) = -1;

  mat U, V;
  vec D;
  svd(U, D, V, sig_pq);

  /*
  cout << U << endl;
  cout << D << endl;
  cout << V << endl;
  */

  mat R = U * S * trans(V);
  //cout << R << endl;
  double s = trace(diagmat(D) * S)/sig_p2;
  vec t = trans(mu_q) - s * R * trans(mu_p);

  R = R * s;
  return make_pair(Matrix2x2<T>(R(0, 0), R(0, 1), R(1, 0), R(1, 1)),
	  Point2<T>(t(0), t(1)));
}

#endif // TRANSFORM_H
