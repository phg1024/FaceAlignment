#ifndef TESTARMADILLO_H
#define TESTARMADILLO_H

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS

#include <armadillo>
using namespace arma;

namespace FATest {
namespace AramdilloTest {

void testArmadillo() {
  cout << "testing armadillo ..." << endl;
  mat A = randu<mat>(5,5);
  vec b = randu<vec>(5);

  vec x = solve(A, b);
  cout << x << endl;

  vec s;
  mat V, U;
  svd(U, s, V, A);
  cout << A << endl;
  cout << U * diagmat(s) * trans(V) << endl;

}

}
}

#endif // TESTARMADILLO_H
