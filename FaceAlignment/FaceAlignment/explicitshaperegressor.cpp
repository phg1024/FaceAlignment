#include "explicitshaperegressor.h"
#include "transform.hpp"

#include "Utils/stringutils.h"
#include "Utils/utility.hpp"

#include <QDomDocument>
#include <QFile>
#include <QStringList>

ExplicitShapeRegressor::ExplicitShapeRegressor()
{
}

void ExplicitShapeRegressor::train(const string &trainSetFile)
{
  QDomDocument doc("trainset");
  QFile file(trainSetFile.c_str());
  if (!file.open(QIODevice::ReadOnly)) {
    cerr << "Failed to open file " << trainSetFile << endl;
    return;
  }
  if (!doc.setContent(&file)) {
    cerr << "Failed to parse file " << trainSetFile << endl;

    file.close();
    return;
  }
  file.close();

  map<string, string> configs;
  vector<int> indices;

  QDomElement docElem = doc.documentElement();
  QDomNode n = docElem.firstChild();
  while(!n.isNull()) {
    QDomElement e = n.toElement(); // try to convert the node to an element.
    if(!e.isNull()) {
      //cout << qPrintable(e.tagName()) << "\t" << qPrintable(e.text()) << endl; // the node really is an element.
      if( e.tagName() == "indices" ) {
        QStringList strlist = e.text().split(" ");
        indices.reserve(strlist.size());
        for(auto str : strlist) {
          indices.push_back(str.toInt());
        }
      }
      else {
        configs[e.tagName().toStdString()] = e.text().toStdString();
      }
    }
    n = n.nextSibling();
  }

  trainWithSamples(configs, indices);
}

void ExplicitShapeRegressor::trainWithSamples(const map<string, string> &configs, const vector<int> &indices) {
  for(auto p : configs) {
    cout << p.first << ": " << p.second << endl;
    if( p.first == "F" ) settings.F = PhGUtils::fromString<int>(p.second);
    else if( p.first == "K" ) settings.K = PhGUtils::fromString<int>(p.second);
    else if( p.first == "P" ) settings.P = PhGUtils::fromString<int>(p.second);
    else if( p.first == "T" ) settings.T = PhGUtils::fromString<int>(p.second);
    else if( p.first == "G" ) settings.G = PhGUtils::fromString<int>(p.second);
    else if( p.first == "beta" ) settings.beta = PhGUtils::fromString<double>(p.second);
  }

  cout << "loading data ..." << endl;

  /// load the images
  loadData(configs, indices);

  /// create a bunch of ferns
  createFerns();

  /// generate initial shapes
  generateInitialShapes();

  /// train the ferns
  trainRegressors();
}

void ExplicitShapeRegressor::generateInitialShapes() {
  // Seed with a real random value, if available
  std::random_device rd;
  std::default_random_engine e1(rd());
  std::uniform_int_distribution<int> uniform_dist(0, data.size()-1);
  settings.N = data.size() * settings.G;
  trainingData.resize(settings.N);
  int j=0;
  for( auto &d : data ) {
    for(int i=0;i<settings.G;++i) {
      int idx = uniform_dist(e1);
      trainingData[j].img = d.img;
      trainingData[j].truth = d.truth;
      trainingData[j].guess = data[idx].truth;
    }
  }
}

void ExplicitShapeRegressor::trainRegressors() {
  /// T stages in total
  for(int i=0;i<settings.T;++i) {
    computeNormalizedShapeTargets();
    learnStageRegressor(i);
    updateGuessShapes();
  }
}

void ExplicitShapeRegressor::computeNormalizedShapeTargets() {
  normalizedShapeTargets.resize(settings.N);
  /// compute mean shape
  /// simply the ground truth of a training shape as the mean shape
  ShapeVector meanShape = trainingData[0].truth;

  for(int i=0;i<settings.N;++i) {
    Matrix3x3<double> M = Transform<double>::estimateTransformMatrix(trainingData[i].guess.pts,
                                                         meanShape.pts);
    normalizedShapeTargets[i].pts = Transform<double>::transform((trainingData[i].truth - trainingData[i].guess).pts, M);
  }
}

void ExplicitShapeRegressor::learnStageRegressor(int stageIdx) {
  regressor_t &R = regressors[stageIdx];

  auto localCoords = generateLocalCoordiantes();
  auto sipixels = extractShapeIndexedPixels(localCoords);
  /// compute pixel-pixel covariance

  /// update all targets
  ShapeVector Y = normalizedShapeTargets[0];
  for(int k=0;k<settings.K;++k) {
    /// select features based on correlation

    /// sample F thresholds from a uniform distribution

    /// partition training samples into 2^F bins

    /// compute outpus of all bins

    /// construct a fern

    /// update the targets
  }
}

vector<ExplicitShapeRegressor::LocalCoordinates> ExplicitShapeRegressor::generateLocalCoordiantes() {
  settings.Nfp = data.front().truth.pts.size();

  vector<ExplicitShapeRegressor::LocalCoordinates> localCoords(settings.P);

  // Seed with a real random value, if available
  std::random_device rd;
  std::default_random_engine e1(rd());
  std::default_random_engine e2(rd());
  std::uniform_real_distribution<double> rand_range(-settings.kappa, settings.kappa);
  std::uniform_int_distribution<int> rand_fp(0, settings.Nfp-1);

  for(int i=0;i<settings.P;++i) {
    LocalCoordinates &lc = localCoords[i];
    lc.fpidx = rand_fp(e1);
    lc.pt.x = rand_range(e2);
    lc.pt.y = rand_range(e2);
  }

  return localCoords;
}

vector<ExplicitShapeRegressor::ShapeIndexedPixel> ExplicitShapeRegressor::extractShapeIndexedPixels(const vector<ExplicitShapeRegressor::LocalCoordinates> &localCoords) {
  vector<ExplicitShapeRegressor::ShapeIndexedPixel> sipixels;

  return sipixels;
}

void ExplicitShapeRegressor::updateGuessShapes() {

}

void ExplicitShapeRegressor::loadData(const map<string, string> &configs, const vector<int> &indices)
{
  string path = configs.at("path");
  string prefix = configs.at("prefix");
  string postfix = configs.at("postfix");
  string imgext = configs.at("imgext");
  string ptsext = configs.at("ptsext");
  int digits = PhGUtils::fromString<int>(configs.at("digits"));

  if( PhGUtils::contains(configs, "startidx") && PhGUtils::contains(configs, "endidx") ) {
    /// use start index and end index to load images
    int startIdx = PhGUtils::fromString<int>(configs.at("startidx"));
    int endIdx = PhGUtils::fromString<int>(configs.at("endidx"));

    data.resize(endIdx - startIdx + 1);
    for(int idx=startIdx, j=0; idx<endIdx;++idx, ++j) {
      /// load image and points
      string imgfile, ptsfile;
      string idxstr = PhGUtils::padWith(PhGUtils::toString(idx), '0', digits);
      imgfile = path + "/" + prefix + idxstr + postfix + imgext;
      ptsfile = path + "/" + prefix + idxstr + postfix + ptsext;
      ImageData &imgdata = data[j];
      imgdata.loadImage(imgfile);
      imgdata.loadPoints(ptsfile);
    }
  }
  else if (!indices.empty()) {
    int j=0;
    data.resize(indices.size());
    for(auto idx : indices) {
      /// load image and points
      string imgfile, ptsfile;
      string idxstr = PhGUtils::padWith(PhGUtils::toString(idx), '0', digits);
      imgfile = path + "/" + prefix + idxstr + postfix + imgext;
      ptsfile = path + "/" + prefix + idxstr + postfix + ptsext;

      ImageData &imgdata = data[j];
      imgdata.loadImage(imgfile);
      imgdata.loadPoints(ptsfile);

      ++j;
    }
  }
  else {
    cerr << "No data to load." << endl;
  }
}

void ExplicitShapeRegressor::createFerns()
{
  regressors.resize(settings.T);
  for(auto stage : regressors) {
    stage.resize(settings.K);
    for(auto fern : stage) {
      fern.resize(settings.F);
    }
  }
}

void ExplicitShapeRegressor::evaluate(const string &testSetFile)
{

}

void ExplicitShapeRegressor::load(const string &filename)
{

}

void ExplicitShapeRegressor::write(const string &filename)
{
  ofstream fout(filename);
  fout << "Regressor" << endl;
  fout.close();
}


void ExplicitShapeRegressor::ImageData::loadImage(const string &filename)
{
  cout << "loading image " << filename << endl;
  img = imread(filename.c_str(), CV_LOAD_IMAGE_UNCHANGED);

  namedWindow(filename.c_str(), CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"

#ifdef FA_DEBUG
  imshow(filename.c_str(), img); //display the image which is stored in the 'img' in the "MyWindow" window
  destroyWindow(filename.c_str());
#endif
}

void ExplicitShapeRegressor::ImageData::loadPoints(const string &filename)
{
  cout << "loading points " << filename << endl;
  ifstream f(filename);
  int npts;
  f >> npts;
  truth.pts.resize(npts);
  for(int i=0;i<npts;++i) {
    f >> truth.pts[i].x >> truth.pts[i].y;
#ifdef FA_DEBUG
    cout << truth.pts[i].toString() << ' ';
#endif
  }
  cout << endl;
  f.close();
}
