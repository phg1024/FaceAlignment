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

template <typename InputType, typename OutputType>
class FernRegressor {
public:
  typedef InputType input_t;
  typedef OutputType output_t;

  FernRegressor(){}
  void resize(size_t size) { N = size; }
  void setThresholds(const InputType &thres) { thresholds = thres; }
  void initialize(const InputType &thres,
                  const vector<OutputType> &out) {
    thresholds = thres;
    outputs = out;
  }

  OutputType evaluate(const InputType &input);
  const InputType& getThresholds() const {
    return thresholds;
  }
  const vector<OutputType>& getOutputs() const {
    return outputs;
  }

  InputType& getThresholds() {
    return thresholds;
  }
  vector<OutputType>& getOutputs() {
    return outputs;
  }

protected:
  int computeBinIndex(const InputType &input);

private:
  size_t N;
  InputType thresholds;
  vector<OutputType> outputs;
};

template <typename InputType, typename OutputType>
OutputType FernRegressor<InputType, OutputType>::evaluate(const InputType &input)
{
  int binIdx = computeBinIndex(input);
  return outputs[binIdx];
}

template <typename InputType, typename OutputType>
int FernRegressor<InputType, OutputType>::computeBinIndex(const InputType &input)
{
  int binIdx = 0;
  for(int i=0;i<N;++i) {
    binIdx <<= 1;
    binIdx += (input(i) < thresholds(i))?0:1;
  }
  return binIdx;
}

#endif


