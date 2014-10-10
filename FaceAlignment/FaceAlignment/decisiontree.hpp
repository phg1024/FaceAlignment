#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "common.h"

template <typename T>
class Data {
public:
    typedef T sample_t;
    Data(){}
    Data(const vector<sample_t> &samples):samples(samples){}
    
    size_t sampleCount() const { return samples.size(); }
    const sample_t& sample(idx_t idx) const{
        assert( idx < samples.size() );
        return samples[idx];
    }
    
private:
    vector<sample_t> samples;
};

template <typename T>
struct TreeNode
{
    typedef T sample_t;
    typedef typename sample_t::item_t item_t;
    
    TreeNode(){}
    TreeNode(const set<int> &dimensions, const set<int> &samples):
    dimensions(dimensions), samples(samples){}
    
    /// split the node with the given dimension and pivot value
    void split(int dim, item_t val) {
        /// @todo need to implement this
    }
    
    /// reference to samples
    bool isLeaf;
    
    int splittingDimension;
    item_t splittingValue;
    
    set<int> dimensions;
    set<int> samples;
    
    shared_ptr<TreeNode<T>> leftChild, rightChild;
};

template <typename T>
class InfomationGainSplitter {
public:
    typedef T sample_t;
    typedef Data<sample_t> data_t;
    typedef TreeNode<sample_t> node_t;
    typedef typename sample_t::item_t item_t;
    
    static bool splitTest(const node_t &node) {
        /// @todo need to implement this
        return false;
    }
    static pair<int, item_t> computeSplittingDimensionAndValue(const node_t &node, const data_t &data) {
        pair<int, item_t> dim_val;
        /// @todo need to implement this
        return dim_val;
    }
};

template <typename SampleType, class NodeType = TreeNode<SampleType>,
          class SplittingStrategy = InfomationGainSplitter<SampleType>>
class DecisionTree
{
public:
    typedef SampleType sample_t;
    typedef typename sample_t::item_t item_t;
    typedef Data<SampleType> data_t;
    typedef NodeType node_t;
    typedef SplittingStrategy strategy_t;
    
    DecisionTree(){}
    
    void train(const data_t &data);
    vector<int> classify(const data_t &data);
    
private:
    shared_ptr<node_t> root;
};

template <typename SampleType, class NodeType, class SplittingStrategy>
void DecisionTree<SampleType, NodeType, SplittingStrategy>::train(const DecisionTree::data_t &data) {
    set<int> dimensions;
    set<int> samples;
    size_t n = data.sampleCount();
    int ndims = data_t::sample_t::item_t::ndims();
    for(int i=0;i<n;++i) { samples.insert(i); }
    for(int i=0;i<ndims;++i) { dimensions.insert(i); }
    
    root = shared_ptr<node_t>(new node_t(dimensions, samples));
    
    queue<shared_ptr<node_t>> Q;
    Q.push(root);

    /// build a decision tree by recursively splitting tree node until convergence is reached
    
    /// build the tree in BFS manner
    while( !Q.empty() ) {
        auto cur = Q.front();
        Q.pop();
        
        /// do we need to further split this node?
        bool needSplit = strategy_t::splitTest(*cur);
        
        if( needSplit ) {
            /// choose a dimension for splitting
            pair<int, item_t> dim_val = strategy_t::computeSplittingDimensionAndValue(*cur, data);
            int sdim = dim_val.first;
            item_t sval = dim_val.second;
            
            /// split the node and repeat the process at those nodes
            cur->split(sdim, sval);
            Q.push(cur->leftChild);
            Q.push(cur->rightChild);
        }
        else {
            /// otherwise, the node is good enough
        }
    }
}

template <typename SampleType, class NodeType, class SplittingStrategy>
vector<int> DecisionTree<SampleType, NodeType, SplittingStrategy>::classify(const DecisionTree::data_t &data) {
    vector<int> clabs;
    return clabs;
}
#endif // DECISIONTREE_H
