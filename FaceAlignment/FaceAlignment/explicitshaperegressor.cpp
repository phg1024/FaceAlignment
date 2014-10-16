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
  settings.w = data.front().img.cols;
  settings.h = data.front().img.rows;
  cout << "image size " << settings.w << "x" << settings.h << endl;

  int j=0;
  cout << "initial error:" << endl;
  for( auto &d : data ) {
    for(int i=0;i<settings.G;++i) {
      int idx = uniform_dist(e1);
      trainingData[j].img = d.img;
      trainingData[j].truth = d.truth;
      trainingData[j].guess = data[idx].truth;
      //cout << j << ", " << idx << ": " << trainingData[j].truth.n_elem << ", " << trainingData[j].guess.n_elem << endl;

      double error = norm(trainingData[j].guess - trainingData[j].truth);
      cout << error << ' ';

      j++;
    }
  }
  cout << endl;
  cout << "total initial shapes = " << j << endl;
}

void ExplicitShapeRegressor::trainRegressors() {
  /// T stages in total
  for(int i=0;i<settings.T;++i) {
    computeNormalizedShapeTargets();
    learnStageRegressor(i);
    updateGuessShapes(i);
  }
}

void ExplicitShapeRegressor::computeNormalizedShapeTargets() {
  cout << "computing normalized shapes ..." << endl;
  normalizedShapeTargets.resize(settings.N);
  normalizationMatrices.resize(settings.N);

  /// compute mean shape
  /// simply the ground truth of a training shape as the mean shape
  ShapeVector meanShape = trainingData[0].truth;

  for(int i=0;i<settings.N;++i) {
    Matrix3x3<double> &M = normalizationMatrices[i];
    M = Transform<double>::estimateTransformMatrix(trainingData[i].guess,
                                                   meanShape);
    normalizedShapeTargets[i] = Transform<double>::transform(trainingData[i].truth - trainingData[i].guess, M);
  }
  cout << "normalized shapes computed." << endl;
}

void ExplicitShapeRegressor::learnStageRegressor(int stageIdx) {
  cout << "learning stage " << stageIdx << "... " << endl;
  regressor_t &R = regressors[stageIdx];
  vector<featureselector_t> &fs = featureSelectors[stageIdx];

  cout << "generating local coordinates ..." << endl;
  auto localCoords = generateLocalCoordiantes();
  cout << "done." << endl;

  cout << "extracting shape indexed pixels ..." << endl;
  auto sipixels = extractShapeIndexedPixels(localCoords); /// rho
  cout << "done." << endl;

  /// compute pixel-pixel covariance
  cout << "computing pixel-pixel covariance ..." << endl;
  mat rho(settings.N, settings.P);
  for(int i=0;i<settings.N;++i) {
    for(int j=0;j<settings.P;++j) {
        rho.at(i, j) = sipixels[i].pixels[j];
    }
  }
  mat covRho = cov(rho);
  cout << "done." << endl;

  cout << "computing matrix Y ..." << endl;
  mat Y(settings.N, settings.Nfp * 2);  /// Y matrix
  for(int i=0;i<settings.N;++i) {
    for(int j=0;j<settings.Nfp*2;++j) {
      Y(i, j) = normalizedShapeTargets[i](j);
    }
  }
  cout << "done." << endl;

  /// update all targets
  cout << "updating all targets ..." << endl;
  for(int k=0;k<settings.K;++k) {
    //cout << "stage " << stageIdx << " fern #" << k << endl;
    auto &fsk = fs[k];
    /// select features based on correlation
    fsk = correlationBasedFeatureSelection(Y, rho, covRho);
    //cout << "features selected ..." << endl;

    /// sample F thresholds from a uniform distribution
    vec thresholds(settings.F, fill::randu);
    thresholds = (thresholds - 0.5) * 0.2 * 255.0;

    /// partition training samples into 2^F bins
    vector<set<int>> bins = partitionSamplesIntoBins(rho, fsk, thresholds);

    /// compute outputs of all bins
    mat outputs = computeBinOutputs(bins);

    /// construct a fern
    vector<vec> outputvec(bins.size());
    for(int rIdx = 0;rIdx < bins.size(); rIdx++) {
      outputvec[rIdx] = trans(outputs.row(rIdx));
    }
    R[k].initialize(thresholds, outputvec);
    //cout << "ferns constructed ..." << endl;

    /// update the targets
    mat updateMat(settings.N, settings.Nfp*2);
    for(int sidx = 0; sidx < settings.N; ++sidx) {
      vec diff_rho(settings.F);
      for(int fidx=0;fidx<settings.F;++fidx) {
        diff_rho(fidx) = fsk[fidx].rho_m(sidx) - fsk[fidx].rho_n(sidx);
      }
      updateMat.row(sidx) = trans(R[k].evaluate(diff_rho));
    }

    Y = Y + updateMat;
  }
  cout << "done." << endl;
}

mat ExplicitShapeRegressor::computeBinOutputs(const vector<set<int>> &bins) {
  mat outputs(bins.size(), 2*settings.Nfp, fill::zeros);

  for(int i=0;i<bins.size();++i) {
    if( bins[i].empty() ) continue;

    for(auto sid : bins[i]) {
      auto &sample = trainingData[sid].truth;
      for(int j=0;j<settings.Nfp*2;++j) {
        outputs(i, j) += sample(j);
      }
    }

    const double factor = 1.0 / (1 + settings.beta * bins[i].size());
    for(int j=0;j<settings.Nfp*2;++j) {
      outputs(i, j) *= factor;
    }
  }
  return outputs;
}

vector<set<int>> ExplicitShapeRegressor::partitionSamplesIntoBins(const mat &rho, const vector<FernFeature> &features, const vec &thresholds) {
  vector<set<int>> bins(1<<settings.F, set<int>());
  for(int i=0;i<settings.N;++i) {
    int binIdx = 0;
    for(int j=0;j<settings.F;++j) {
      binIdx <<= 1;

      auto &f = features[j];
      double diff = rho(i, f.m) - rho(i, f.n);
      binIdx += (diff < thresholds(j))?0:1;
    }
    bins[binIdx].insert(i);
  }
  return bins;
}

vector<ExplicitShapeRegressor::FernFeature> ExplicitShapeRegressor::correlationBasedFeatureSelection(const mat &Y, const mat &rho, const mat &covRho) {
  //cout << "selecting features ..." << endl;
  vector<ExplicitShapeRegressor::FernFeature> features(settings.F);
  for(int i=0;i<settings.F;++i) {
    vec nu(2*settings.Nfp, fill::randn);
    vec Yprob = Y * nu; /// N-by-1 vector

    vec covYprob_rho = trans(cov(Yprob, rho));
    double varYprob = var(Yprob);

    auto &f = features[i];

    f.m = 0;
    f.n = 0;
    f.rho_m = rho.col(0);
    f.rho_n = rho.col(0);
    f.coor_rhoDiff = 0;

    for(int m=0;m<settings.P;++m) {
      for(int n=0;n<settings.P;++n) {
        double varRhoDRho = covRho(m, m) + covRho(n, n) - 2.0 * covRho(m, n);
        double corrYprob_rhoDrho = (covYprob_rho(m) - covYprob_rho(n)) / sqrt(varYprob * varRhoDRho);
        if( corrYprob_rhoDrho > f.coor_rhoDiff ) {
          f.coor_rhoDiff = corrYprob_rhoDrho;
          f.rho_m = rho.col(m);
          f.rho_n = rho.col(n);
          f.m = m;
          f.n = n;
        }
      }
    }
  }

  //cout << "done." << endl;
  return features;
}

vector<ExplicitShapeRegressor::LocalCoordinates> ExplicitShapeRegressor::generateLocalCoordiantes() {
  settings.Nfp = data.front().truth.n_elem / 2;
  localCoords.resize(settings.P);

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

vector<ExplicitShapeRegressor::ShapeIndexedPixels> ExplicitShapeRegressor::extractShapeIndexedPixels(const vector<ExplicitShapeRegressor::LocalCoordinates> &localCoords) {
  sipixels.resize(settings.N);

  /// for each shape, extract P pixels
  for(int i=0;i<settings.N;++i) {
    auto &sipixel = sipixels[i];
    sipixel.shapeIdx = i;
    for(int j=0;j<settings.P;++j) {
      Matrix3x3<double> &M = normalizationMatrices[i];
      Point2<double> dp = M * localCoords[j].pt;
      double pos_x = trainingData[i].guess(localCoords[j].fpidx*2) + dp.x;
      double pos_y = trainingData[i].guess(localCoords[j].fpidx*2+1) + dp.y;
      int c = round(pos_x);
      int r = round(pos_y);

      unsigned char p = 0;
      if( r >= settings.h || r < 0 || c >= settings.w || c < 0 ) {
        // not a valid pixel
      }
      else {
        p = trainingData[i].img.at<unsigned char>(r, c);
      }

      sipixel.pixels.push_back(p);
    }

  }
  return sipixels;
}

void ExplicitShapeRegressor::updateGuessShapes(int stageIdx) {
  cout << "updating guess shapes ..." << endl;
  vector<double> error(settings.N, 0);
  for(int i=0;i<settings.N;++i) {
    auto &M = normalizationMatrices[i];
    auto inv_M = M.inv();
    ShapeVector deltaShape = applyStageRegressor(i, stageIdx);
    trainingData[i].guess += Transform<double>::transform(deltaShape, inv_M);

    // compute error
    error[i] = norm(trainingData[i].guess - trainingData[i].truth);
    cout << error[i] << ' ';
  }
  cout << endl;
}

ExplicitShapeRegressor::ShapeVector ExplicitShapeRegressor::applyStageRegressor(
    int sidx, int stageIdx) {
  ShapeVector deltaS(settings.Nfp*2, fill::zeros);

  vec rho(settings.F);
  for(int k=0;k<settings.K;++k) {
    for(int fidx=0;fidx<settings.F;++fidx) {
      int m = featureSelectors[stageIdx][k][fidx].m;
      int n = featureSelectors[stageIdx][k][fidx].n;
      rho(fidx) = sipixels[sidx].pixels[m] - sipixels[sidx].pixels[n];
    }
    vec ds = regressors[stageIdx][k].evaluate(rho);
    deltaS += ds;
  }
  return deltaS;
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
  featureSelectors.resize(settings.T);

  for(auto &stage : regressors) {
    stage.resize(settings.K);
    for(auto &fern : stage) {
      fern.resize(settings.F);
    }
  }

  for(auto &fselector : featureSelectors) {
    fselector.resize(settings.K);
    for(auto &fs : fselector) {
      fs.resize(settings.F);
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
  cout << "image size = " << img.cols << "x" << img.rows << endl;

#ifdef FA_DEBUG
  namedWindow(filename.c_str(), CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
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
  truth = vec(npts*2);
  for(int i=0;i<npts;++i) {
    f >> truth(i*2) >> truth(i*2+1);
#ifdef FA_DEBUG
    cout << truth.pts[i].toString() << ' ';
#endif
  }
#ifdef FA_DEBUG
  cout << endl;
#endif
  cout << "points = " << npts << endl;
  f.close();
}
