//
//  Header.h
//  FaceAlignment
//
//  Created by Peihong Guo on 10/10/14.
//  Copyright (c) 2014 Peihong Guo. All rights reserved.
//

#ifndef FaceAlignment_Header_h
#define FaceAlignment_Header_h

#include "fernregressor.hpp"


namespace FATest {
  namespace FernTest {
    struct InputType {
      InputType(){}
      InputType(int x, int y, int z){
        f[0] = x; f[1] = y; f[2] = z;
      }
      int operator()(int idx) const{
        return f[idx];
      }
      int f[3];
    };
    
    typedef double OutputType;
    typedef pair<InputType, OutputType> SampleType;
    
    OutputType groundTruth(const InputType &input) {
      return input(0) * 4 + input(1) * 2 + input(2) + rand() / (double)RAND_MAX - 0.5;
    }
    
    struct Estimator {
      static double estimate(const set<int> &sampleIndices, const vector<SampleType> &samples, const vector<double> &guess) {
        double val = 0;
        for(auto sidx : sampleIndices) {
          val += (samples[sidx].second - guess[sidx]);
        }
        if( !sampleIndices.empty() )
          val /= sampleIndices.size();
        return val;
      }
    };
    
    vector<SampleType> generateSamples(int n) {
      // Seed with a real random value, if available
      std::random_device rd;
      
      // Choose a random mean between 1 and 6
      std::default_random_engine e1(rd());
      std::uniform_int_distribution<int> uniform_dist(0, 1);
      
      vector<SampleType> samples;
      for(int i=0;i<n;++i) {
        double x = uniform_dist(e1);
        double y = uniform_dist(e1);
        double z = uniform_dist(e1);
        InputType input(x, y, z);
        samples.push_back(make_pair(input, groundTruth(input)));
      }
      return samples;
    }
    
    void testFerns() {
      typedef FernRegressor<InputType, OutputType, Estimator> fern_t;
      int T = 4;  // number of stages
      int K = 20; // number of ferns in each stage
      typedef vector<fern_t> stage_t;
      vector<stage_t> stages(T);
      for(auto &stage : stages) {
        stage.resize(K);
        for(auto &fern : stage) {
          fern.resize(3);
        }
      }
      
      const int nTraingSamples = 10;
      const int nTestSamples = 2;
      vector<SampleType> trainingSet = generateSamples(nTraingSamples);
      vector<double> guess(nTraingSamples, 3.5);
      vector<SampleType> testSet = generateSamples(nTestSamples);
      
      for(int i=0;i<T;++i) {
        for(int j=0;j<K;++j) {
          auto &fern = stages[i][j];
          //cout << "training fern " << i << ", " << j << endl;
          fern.train(trainingSet, guess);
          
          // update guess
          auto delta = fern.evaluate(trainingSet);
          //cout << "j = " << j << endl;
          for(int k=0;k<guess.size();++k) {
            //cout << guess[k] - trainingSet[k].second << ", ";
            guess[k] += delta[k];
          }
          //cout << endl;
        }
      }

      vector<double> deltaval;
      vector<double> result(nTestSamples, 3.5);
      for(int i=0;i<T;++i) {
        for(int j=0;j<K;++j) {
          auto &fern = stages[i][j];
          //cout << "evaluating fern " << i << ", " << j << endl;
          deltaval = fern.evaluate(testSet);
          for(int k=0;k<result.size();++k) {
            //cout << deltaval[k] << ' ';
            result[k] += deltaval[k];
          }
          //cout << endl;
        }
      }
      double sum=0.0;
      for(int i=0;i<testSet.size();++i) {
        cout << result[i] << " vs " << testSet[i].second << endl;
        double diff = result[i] - testSet[i].second;
        sum += diff*diff;
      }
      cout << "RMSE = " << sum / testSet.size() << endl;
      
    }
  }
}

#endif
