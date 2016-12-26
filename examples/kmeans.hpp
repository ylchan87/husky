#pragma once

#include <cfloat>
#include <memory>
#include <core/engine.hpp>
#include "lib/aggregator_factory.hpp"

namespace husky {

template<typename ObjT, typename FeatureT>
class Kmeans{
public:
    // interface
    void run();
    int getClass(const ObjT&);

    std::vector<FeatureT> clusterCenter;

    Kmeans() = default;

    //no init centers provided
    Kmeans(ObjList<ObjT>& obj_list, 
            std::function<FeatureT(const ObjT&)> get_feature,
            std::function<double(const FeatureT&, const FeatureT&)> get_distance,
            const int k,
            const int iter
            ): obj_list(obj_list), get_feature(get_feature), get_distance(get_distance), k(k), iter(iter), clusterCenter(){}

    //init centers provided
    Kmeans(ObjList<ObjT>& obj_list, 
            std::function<FeatureT(const ObjT&)> get_feature,
            std::function<double(const FeatureT&, const FeatureT&)> get_distance,
            std::vector<FeatureT> clusterCenter, // initial center
            const int iter
            ): obj_list(obj_list), get_feature(get_feature), get_distance(get_distance), k(0), iter(iter), clusterCenter(clusterCenter){}

private:
    ObjList<ObjT>& obj_list;
    std::function<FeatureT(const ObjT&)> get_feature; 
    std::function<double(const FeatureT&, const FeatureT&)> get_distance; 
    int k;
    int iter;
    int feature_num;
    
    std::vector<FeatureT>& init() {
        husky::base::log_info("init ");
        if (clusterCenter.size() == 0) { // init_center not provided
            //an agg that concate vectors
            husky::lib::Aggregator< std::vector<FeatureT> > init_center_agg(
              std::vector<FeatureT>(),
              [](std::vector<FeatureT>& a, const std::vector<FeatureT>& b) {
                a.insert(a.end(), b.begin(), b.end());
              },
              [](std::vector<FeatureT>& v) {
                v = std::move(std::vector<FeatureT>());
            });

            int worker_num = husky::Context::get_worker_info().get_num_workers();
            int nSelect = k/worker_num + 1;
            std::vector<FeatureT> local_init_center;
            for (int i=0;i<nSelect;i++){
              local_init_center.push_back( get_feature(obj_list.get_data()[i]));
            }
            init_center_agg.update( local_init_center );
            husky::lib::AggregatorFactory::sync();

            clusterCenter = init_center_agg.get_value();
            clusterCenter.resize(k);
        }
        else { // init_center provided
            assert( clusterCenter[0].size() == get_feature(obj_list.get_data()[0]).size() );
            k = clusterCenter.size();
        }
        feature_num = clusterCenter[0].size();
        return clusterCenter;
    }

};


template<typename ObjT, typename FeatureT>
void Kmeans<ObjT, FeatureT>::run() {
    init();
        husky::base::log_info("run ");

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    std::vector<husky::lib::Aggregator<FeatureT>> clusterCenterAggVec;
    std::vector<husky::lib::Aggregator<int>> clusterCountAggVec;
    for (int i=0;i<k;i++){
      clusterCenterAggVec.push_back( husky::lib::Aggregator<FeatureT>() );
      clusterCountAggVec.push_back( husky::lib::Aggregator<int>() );
    }

    // list_execute
    for (int it = 0; it < iter; ++ it) {
        // if (worker.id == 0) {
        //     for (int i = 0; i < k; ++ i) {
        //         std::cout << "center: " << i << " : ";
        //         printVector(feature_vector[i]);
        //     }
        // }

        // assign obj to centers
        husky::base::log_info("iter " + std::to_string(it) + "start ");
        list_execute(obj_list, {}, {&ac}, [&](ObjT& obj){
            // get idx of nearest center
            int idx = getClass(obj);
            // assgin
            clusterCenterAggVec[idx].update( get_feature(obj) );
            clusterCountAggVec[idx].update(1);
        });

        //update cluster center
        for (int i = 0; i < k; ++ i) {
            int count = clusterCountAggVec[i].get_value();
            if (count != 0)  clusterCenter[i] = clusterCenterAggVec[i].get_value() / count;
            // printVector(feature_vector[i]);

            clusterCenterAggVec[i].to_reset_each_iter();
            clusterCountAggVec[i].to_reset_each_iter();
        }
    }
    return;
}

template<typename ObjT, typename FeatureT>
int Kmeans<ObjT, FeatureT>::getClass(const ObjT& obj) {
    double dist = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < k; ++ i) {
        double d = get_distance( get_feature(obj), clusterCenter[i]);
        if (d < dist) {
            dist = d;
            idx = i;
        }
    }
    return idx;
}

/*
Usage:
Husky::kmeans(point_list, 
    init_center,
    [](Point& p){ return p.feature;},
    2 // iter nums
);
Husky::kmeans<std::vector<double>>(point_list, 
    [](Point& p){ return p.feature;},
    4, // center nums
    2 // iter nums
);

*/
};

