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

namespace FATest {
    
template <int n>
struct ItemType{
    typedef double value_t;
    ItemType(){}
    ItemType(double x, double y){ f[0] = x; f[1] = y; }
    static int ndims() { return n; }
    
    double f[n];
};

template <typename DType>
struct SampleType {
    typedef DType item_t;
    SampleType(){}
    SampleType(const item_t &data, int clabel=-1):data(data), clabel(clabel){}
    item_t data;
    int clabel;
};
    
typedef ItemType<2> item_t;
typedef SampleType<item_t> sample_t;

vector<sample_t> generateSamples(int n) {
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
    return samples;
}

void testDecisionTree() {
    typedef Data<sample_t> data_t;
    Data<sample_t> trainingSet(generateSamples(250));
    Data<sample_t> testSet(generateSamples(50));
    
    DecisionTree<sample_t> dt;
    dt.train(trainingSet);
    vector<int> result = dt.classify(testSet);
    int correctCount = 0;
    for(int i=0;i<result.size();++i) {
        cout << result[i] << " vs " << testSet.sample(i).clabel << endl;
        correctCount += (result[i] == testSet.sample(i).clabel)?1:0;
    }
    cout << "correctness ratio = " << correctCount / (double) result.size() << endl;
}
    
}

#endif
