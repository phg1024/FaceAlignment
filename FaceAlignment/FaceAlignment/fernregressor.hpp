//
//  fernregressor.hpp
//  FaceAlignment
//
//  Created by Peihong Guo on 10/10/14.
//  Copyright (c) 2014 Peihong Guo. All rights reserved.
//

#ifndef FaceAlignment_fernregressor_hpp
#define FaceAlignment_fernregressor_hpp

#include "common.h"

template <typename InputType, typename OutputType, class Estimator, int N>
class FernRegressor {
public:
  typedef InputType input_t;
  typedef OutputType output_t;
  typedef pair<input_t, output_t> sample_t;
  
  FernRegressor();
  
  void train(const vector<sample_t> &trainingSet, const vector<output_t> &currentGuess);
  output_t evaluate(const input_t &input);
  vector<output_t> evaluate(const vector<sample_t> &testSet);
  
protected:
  idx_t computeBinIndex(const input_t &input);
  
private:
  vector<output_t> bins;
  
};

template <typename InputType, typename OutputType, class Estimator, int N>
FernRegressor<InputType, OutputType, Estimator, N>::FernRegressor() {
  bins.resize(1<<N);
}

template <typename InputType, typename OutputType, class Estimator, int N>
void FernRegressor<InputType, OutputType, Estimator, N>::train(const vector<FernRegressor::sample_t> &trainingSet, const vector<output_t> &guess) {
  vector<set<int>> histogram(bins.size());
  
  /// distribute all training samples to the bins
  for(int i=0;i<trainingSet.size();++i) {
    auto &sin = trainingSet[i].first;
    idx_t binIdx = computeBinIndex(sin);
    histogram[binIdx].insert(i);
  }
  
  /// compute the output of each bin using the estimator
  for(int i=0;i<bins.size();++i) {
    bins[i] = Estimator::estimate(histogram[i], trainingSet, guess);
  }
}

template <typename InputType, typename OutputType, class Estimator, int N>
idx_t FernRegressor<InputType, OutputType, Estimator, N>::computeBinIndex(const FernRegressor::input_t &input) {
  idx_t binIdx = 0;
  for(int i=0;i<N;++i) {
    binIdx <<= 1;
    binIdx += input(i);
  }
  return binIdx;
}

template <typename InputType, typename OutputType, class Estimator, int N>
typename FernRegressor<InputType, OutputType, Estimator, N>::output_t FernRegressor<InputType, OutputType, Estimator, N>::evaluate(const FernRegressor::input_t &input) {
  idx_t binIdx = computeBinIndex(input);
  return bins[binIdx];
}

template <typename InputType, typename OutputType, class Estimator, int N>
vector<typename FernRegressor<InputType, OutputType, Estimator, N>::output_t> FernRegressor<InputType, OutputType, Estimator, N>::evaluate(const vector<FernRegressor::sample_t> &testSet) {
  vector<output_t> results(testSet.size());
  for(int i=0;i<testSet.size();++i) {
    results[i] = evaluate(testSet[i].first);
  }
  return results;
}
#endif
