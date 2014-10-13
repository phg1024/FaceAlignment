#ifndef EXPLICITSHAPEREGRESSOR_H
#define EXPLICITSHAPEREGRESSOR_H

#include "common.h"

#include "fernregressor.hpp"
#include "Geometry/point.hpp"

#include "opencv2/highgui/highgui.hpp"
using namespace cv;

class ExplicitShapeRegressor
{
public:
  ExplicitShapeRegressor();

  void train(const string &trainSetFile);
  void evaluate(const string &testSetFile);

  void load(const string &filename);
  void write(const string &filename);

private:
  struct FernInput {
    vector<int> f;
  };
  typedef PhGUtils::Point2<double> point_t;
  struct ShapeVector {
    ShapeVector& operator+=(const ShapeVector &other) {
      for(size_t i=0;i<pts.size();++i) {
        pts[i] += other.pts[i];
      }
      return (*this);
    }

    ShapeVector operator-(const ShapeVector &other) {
      ShapeVector s = (*this);
      for(size_t i=0;i<pts.size();++i) {
        s.pts[i] -= other.pts[i];
      }
      return s;
    }

    ShapeVector& operator/=(double factor) {
      for(size_t i=0;i<pts.size();++i) {
        pts[i] /= factor;
      }
      return (*this);
    }

    vector<point_t> pts;
  };

  class ShapeEstimator{

  };

  struct LocalCoordinates {
    point_t pt;
    int fpidx;
  };

  struct ShapeIndexedPixel {

  };

  typedef FernRegressor<FernInput, ShapeVector, ShapeEstimator> fern_t;
  typedef vector<fern_t> regressor_t;

protected:
  void trainWithSamples(const map<string, string> &configs, const vector<int> &indices = vector<int>());
  void loadData(const map<string, string> &configs, const vector<int> &indices = vector<int>());
  void createFerns();
  void generateInitialShapes();
  void trainRegressors();
  void computeNormalizedShapeTargets();
  void learnStageRegressor(int stageIdx);
  void updateGuessShapes();
  vector<LocalCoordinates> generateLocalCoordiantes();
  vector<ShapeIndexedPixel> extractShapeIndexedPixels(const vector<LocalCoordinates> &localCoords);
private:

  struct RegressorSetting {
    RegressorSetting():F(5), K(500), P(400), T(10), G(20), beta(1000.0), kappa(10.0){}
    int F, K, P, T, G, N, Nfp;
    double beta, kappa;
  };

  struct ImageData {
    void loadImage(const string &filename);
    void loadPoints(const string &filename);
    Mat img;
    ShapeVector truth;
    ShapeVector guess;
  };

  vector<ImageData> data;
  vector<ImageData> trainingData;
  vector<ShapeVector> normalizedShapeTargets;

  RegressorSetting settings;
  vector<regressor_t> regressors;
};

#endif // EXPLICITSHAPEREGRESSOR_H
