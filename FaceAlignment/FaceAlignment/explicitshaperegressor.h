#ifndef EXPLICITSHAPEREGRESSOR_H
#define EXPLICITSHAPEREGRESSOR_H

#include "common.h"

#include "fernregressor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"

#include "numerical.hpp"

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
  typedef PhGUtils::Point2<double> point_t;
  typedef arma::vec ShapeVector;

  struct LocalCoordinates {
    point_t pt;
    int fpidx;
  };

  struct ShapeIndexedPixels {
    int shapeIdx;
    vector<unsigned char> pixels;
  };

  struct ImageData {
    void loadImage(const string &filename);
    void loadPoints(const string &filename);
	void show();
    cv::Mat img;
    ShapeVector truth;
    ShapeVector guess;
  };

  struct FernFeature {
    double coor_rhoDiff;
    vec rho_m, rho_n;
    int m, n;   /// feature point indices
  };

  typedef FernRegressor<vec, vec> fern_t;
  typedef vector<fern_t> regressor_t;

protected:
  void trainWithSamples(const map<string, string> &configs, const vector<int> &indices = vector<int>());
  void loadData(const map<string, string> &configs, const vector<int> &indices = vector<int>());
  void createFerns();
  void generateInitialShapes();
  void trainRegressors();
  void computeNormalizedShapeTargets();
  void learnStageRegressor(int stageIdx);
  void updateGuessShapes(int stageIdx);
  vector<LocalCoordinates> generateLocalCoordiantes();
  vector<ShapeIndexedPixels> extractShapeIndexedPixels(const vector<LocalCoordinates> &localCoords);
  vector<ExplicitShapeRegressor::FernFeature> correlationBasedFeatureSelection(const mat &Y, const mat &rho, const mat &covRho);
  vector<set<int> > partitionSamplesIntoBins(const mat &rho, const vector<FernFeature> &features, const vec &thresholds);
  mat computeBinOutputs(const vector<set<int> > &bins);
  ExplicitShapeRegressor::ShapeVector applyStageRegressor(int sidx, int stageIdx);

private:

  struct RegressorSetting {
    RegressorSetting():F(5), K(500), P(400), T(10), G(20), beta(1000.0), kappa(10.0){}
    int F, K, P, T, G, N, Nfp;
    int w, h;
    double beta, kappa;
  };

  vector<ImageData> data;
  vector<ImageData> trainingData;
  vector<ShapeVector> normalizedShapeTargets;
  vector<PhGUtils::Matrix2x2<double>> normalizationMatrices;
  vector<ExplicitShapeRegressor::LocalCoordinates> localCoords;
  vector<ExplicitShapeRegressor::ShapeIndexedPixels> sipixels;
  typedef vector<FernFeature> featureselector_t;
  vector<vector<featureselector_t>> featureSelectors;

  RegressorSetting settings;
  vector<regressor_t> regressors;
};

#endif // EXPLICITSHAPEREGRESSOR_H
