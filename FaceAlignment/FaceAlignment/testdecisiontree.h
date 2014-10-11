//
//  testdecisiontree.h
//  FaceAlignment
//
//  Created by Peihong Guo on 10/9/14.
//  Copyright (c) 2014 Peihong Guo. All rights reserved.
//

#ifndef FaceAlignment_testdecisiontree_h
#define FaceAlignment_testdecisiontree_h

#include "decisiontree.hpp"
#include "randomforest.hpp"

namespace FATest {
  namespace DecisionTreeTest {
    
    template <int n>
    struct ItemType{
      typedef double value_t;
      ItemType(){}
      ItemType(double x, double y){ f[0] = x; f[1] = y; }
      static int ndims() { return n; }
      
      double f[n];
    };
    
    template <int N>
    ostream& operator<<(ostream &os, const ItemType<N> &item) {
      for(int i=0;i<N;++i) {
        os << item.f[i] << (i==N-1?"":", ");
      }
      return os;
    }
    
    template <typename DType>
    struct SampleType {
      typedef DType item_t;
      SampleType(){}
      SampleType(const item_t &data, int clabel=-1):data(data), clabel(clabel){}
      item_t data;
      int clabel;
    };
    
    template <typename DType>
    ostream& operator<<(ostream &os, const SampleType<DType> &s) {
      os << s.data << " @ " << s.clabel;
      return os;
    }
    
    typedef ItemType<2> item_t;
    typedef SampleType<item_t> sample_t;
    
    pair<vector<sample_t>, int> generateSamples(int n) {
      // Seed with a real random value, if available
      std::random_device rd;
      
      // Choose a random mean between 1 and 6
      std::default_random_engine e1(rd());
      std::uniform_real_distribution<double> uniform_dist(-10, 10);
      
      vector<sample_t> samples;
      const double px = 0.0, py = 0.0;
      for (int i=0; i<n; ++i) {
        double x = uniform_dist(e1);
        double y = uniform_dist(e1);
        int clabel = 0;
        if( x > px ) clabel += 1;
        if( y > py ) clabel += 2;
        cout << x << ',' << y << " @ " << clabel << endl;
        samples.push_back(sample_t(item_t(x, y), clabel));
      }
      return make_pair(samples, 4);
    }
    
    void resultStats(vector<int> result, const Data<sample_t> &data) {
      int correctCount = 0;
      for(int i=0;i<result.size();++i) {
        //cout << result[i] << " vs " << testSet.sample(i).clabel << endl;
        correctCount += (result[i] == data.sample(i).clabel)?1:0;
      }
      cout << "correctness ratio = " << correctCount / (double) result.size() << endl;
    }
    
    void testDecisionTree() {
      typedef Data<sample_t> data_t;
      auto trainingSamples = generateSamples(200);
      Data<sample_t> trainingSet(trainingSamples.first, trainingSamples.second);
      auto testSamples = generateSamples(50000);
      Data<sample_t> testSet(testSamples.first, testSamples.second);
      
      DecisionTree<sample_t> dt;
      DecisionTree<sample_t, GiniImpuritySplitter<sample_t>> dt2;
      RandomForest<DecisionTree<sample_t>> forest(20);
      RandomForest<DecisionTree<sample_t, GiniImpuritySplitter<sample_t>>> forest2(20);
      
      dt.train(trainingSet);
      dt2.train(trainingSet);
      forest.train(trainingSet);
      forest2.train(trainingSet);
      
      vector<int> result = dt.classify(testSet);
      vector<int> result2 = dt2.classify(testSet);
      vector<int> result3 = forest.classify(testSet);
      vector<int> result4 = forest2.classify(testSet);
      
      resultStats(result, testSet);
      resultStats(result2, testSet);
      resultStats(result3, testSet);
      resultStats(result4, testSet);
    }
    
  }
  
}

#endif
