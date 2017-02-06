// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cfloat>
#include <core/engine.hpp>
#include <memory>
#include <vector>
#include "base/exception.hpp"
#include "lib/aggregator_factory.hpp"

namespace husky {
namespace lib {
namespace ml {
namespace kmeans {

enum class KmeansOpts { kInitSimple, kInitKmeansPP };
namespace KmeansUtil{
    //primary template
    template<typename FeatureVec>
    std::function<double(const FeatureVec&, const FeatureVec&)> get_default_distance_sq_func(FeatureVec*) {
        return nullptr;
    }

    template<typename FeatureVec>
    FeatureVec get_zero(int dim) {
        return FeatureVec(0);
    }

    //specialize for lib::Vector
    using Vec = husky::lib::VectorXd;
    using SpVec = husky::lib::SparseVectorXd;

    template<>
    std::function<double(const Vec&, const Vec&)> get_default_distance_sq_func(Vec*) {
        return [](const Vec& v1, const Vec& v2){ return (v1-v2).squaredNorm();};
    }

    template<>
    std::function<double(const SpVec&, const SpVec&)> get_default_distance_sq_func(SpVec*) {
        return [](const SpVec& v1, const SpVec& v2){ return (v1-v2).squaredNorm();};
    }

    template<>
    Vec get_zero(int dim) {
        return Vec::Zero(dim);
    }
    
    template<>
    SpVec get_zero(int dim) {
        return SpVec(dim);
    }
    
}

template <typename FeatureVec>
class Kmeans {
   public:
    Kmeans(int k =1, int iter =1);

    template <typename ObjT>
    void fit(ObjList<ObjT>& obj_list, husky::AttrList<ObjT,FeatureVec>& feature_col);

    int getClass(const FeatureVec&);

    Kmeans<FeatureVec>& set_k(int k){ this->k = k; return *this;}
    Kmeans<FeatureVec>& set_iter(int iter){ this->iter = iter; return *this;}
    Kmeans<FeatureVec>& set_feature_dim(int dim){ this->dim = dim; return *this;}
    Kmeans<FeatureVec>& set_init_opt(KmeansOpts opt){ this->kInitOption = opt; return *this;}

    Kmeans<FeatureVec>& set_distance_sq_func(std::function<double(const FeatureVec& v1, const FeatureVec& v2)> f){
        this->get_distance_sq = f;
        return *this;
    }

    Kmeans<FeatureVec>& set_centers(std::vector<FeatureVec> clusterCenters){
      this->clusterCenters = clusterCenters;
      return *this;
    }

    const std::vector<FeatureVec>& get_centers(){
      if (!trained) throw base::HuskyException("Kmeans not yet trained");
      return this->clusterCenters;
    }

   private:
    bool trained;

    int k;
    int iter;
    int dim;    
    KmeansOpts kInitOption;

    std::vector<FeatureVec> clusterCenters;

    template <typename ObjT>
    void init(ObjList<ObjT>& obj_list, AttrList<ObjT,FeatureVec>& feature_col);

    template <typename ObjT, typename ColT>
    std::vector<ColT> get_random_rows_col_val(ObjList<ObjT>& obj_list, AttrList<ObjT,ColT>& target_col,
                                              int count, std::function<double()>& dice, AttrList<ObjT,double>* weight_col = NULL);

    std::function<double(const FeatureVec& v1, const FeatureVec& v2)> get_distance_sq;

};

template <typename FeatureVec>
Kmeans<FeatureVec>::Kmeans(int k, int iter) {
    this->k = k;
    this->iter = iter;
    this->kInitOption = KmeansOpts::kInitKmeansPP;
}

template <typename FeatureVec>
template <typename ObjT>
void Kmeans<FeatureVec>::init(ObjList<ObjT>& obj_list, AttrList<ObjT,FeatureVec>& feature_col){
    if (clusterCenters.size() == 0) {  // init_center not provided
        int worker_num = husky::Context::get_worker_info().get_num_workers();

        std::default_random_engine generator;
        generator.seed(1234);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        std::function<double()> dice = std::bind(distribution, generator);

        if (kInitOption == KmeansOpts::kInitSimple) {
            clusterCenters = get_random_rows_col_val( obj_list, feature_col, k, dice);
        } else if (kInitOption == KmeansOpts::kInitKmeansPP) {
            auto& d2_col = obj_list.template create_attrlist<double>("__distanceSq");

            clusterCenters = get_random_rows_col_val( obj_list, feature_col, 1, dice);
            husky::LOG_I << clusterCenters.size() << " got it";

            husky::LOG_I << "first center pick";
            list_execute(obj_list, [&](ObjT& obj) {
                double d2 = get_distance_sq(feature_col.get(obj), clusterCenters.back());
                d2_col.set(obj, d2);
            });

            // pick remaining centers by prob weighted by squared distance to nearest chosen centers
            for (int i = 1; i < k; i++) {
                husky::LOG_I << i << "  center pick";
                auto newClusterCenters = get_random_rows_col_val( obj_list, feature_col, 1, dice, &d2_col);

                husky::LOG_I << newClusterCenters.size() << " then update w";

                list_execute(obj_list, [&](ObjT& obj) {
                    double oldd2 = d2_col.get(obj);
                    double d2 = get_distance_sq(feature_col.get(obj), newClusterCenters.back());
                    d2_col.set(obj, std::min(oldd2,d2));
                });

                clusterCenters.insert( clusterCenters.end(),
                                       std::make_move_iterator(newClusterCenters.begin()),
                                       std::make_move_iterator(newClusterCenters.end()));

            }  // end for loop over i= 1 to k
        }      // end if ( kInitOption == kInitKmeansPP)
    } else {   // init_center provided
        if(clusterCenters[0].size() != dim){
            throw husky::base::HuskyException("User provided centers feature dim differs from Kmeans expectation");
        }
        k = clusterCenters.size();
    }
    for (auto aCenter : clusterCenters){
        husky::LOG_I << "center " << aCenter << std::endl;
    }
}

template <typename FeatureVec>
template <typename ObjT, typename ColT>
std::vector<ColT> Kmeans<FeatureVec>::get_random_rows_col_val(ObjList<ObjT>& obj_list, AttrList<ObjT,ColT>& target_col,
                                                              int count, std::function<double()>& dice, AttrList<ObjT,double>* weight_col){
    // all datapoints are laid on a line, each span length equal to its weight
    // a point is randomly chosen on the line, the datapoint that this point falls on is picked
    husky::LOG_I << "random at start " << dice() << std::endl;

    int worker_num = husky::Context::get_worker_info().get_num_workers();
    int tid = husky::Context::get_global_tid();

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    std::vector<husky::lib::Aggregator<double>> totWs;
    for (int i = 0; i < worker_num; i++) {
        totWs.push_back(husky::lib::Aggregator<double>());
    }

    // an agg that concate vectors
    husky::lib::Aggregator<std::vector<FeatureVec>> selected_rows_col_val(
        std::vector<FeatureVec>(),
        [](std::vector<FeatureVec>& a, const std::vector<FeatureVec>& b) {
            a.insert(a.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
        },
        [](std::vector<FeatureVec>& v) { v = std::move(std::vector<FeatureVec>()); });

    if (weight_col==NULL){
        totWs[tid].update( obj_list.get_size());
        husky::lib::AggregatorFactory::sync();
    }else{
        list_execute(obj_list, {}, {&ac}, [&](ObjT& obj) {
            totWs[tid].update(weight_col->get(obj));
        });
    }

    double totW = 0.0;
    double preceedW = 0.0;
    for (int i=0;i<worker_num;++i) {
        totW += totWs[i].get_value();
        if (i<tid) preceedW += totWs[i].get_value();
        husky::LOG_I << "Ws " << i << " " << totWs[i].get_value();
    }

    husky::LOG_I << "totW " << totW;
    husky::LOG_I << "preceedW " << preceedW;

    std::list<double> targetWs;
    for (int i=0;i<count; ++i){
        double targetW = totW * dice();
        husky::LOG_I << "targetW " << targetW;
        if (targetW>preceedW && targetW<(preceedW+totWs[tid].get_value())){
            targetWs.push_back(targetW);
            husky::LOG_I << "targetW added" << targetW;
        }
    }
    targetWs.sort();

    double currentW = preceedW;
    double targetW;

    if (weight_col==NULL){
        int preceedIdx = (int)round(preceedW);
        while(true){
            if (targetWs.size()==0) break;
            int targetIdx = (int)(targetWs.front() - preceedW);
            targetWs.pop_front();
            if (targetIdx<0) targetIdx=0;
            if (targetIdx>=obj_list.get_size()) targetIdx=obj_list.get_size()-1;
            selected_rows_col_val.update( {target_col[targetIdx]} );
            husky::LOG_I << "ran pick idx " << targetIdx;
        }
    }else{
        if (targetWs.size()==0) goto random_pick_done;
        targetW = targetWs.front();
        targetWs.pop_front();
        //loop over local objs
        for (auto& obj : obj_list.get_data()) {
            currentW += weight_col->get(obj);
            while (targetW<currentW){
                selected_rows_col_val.update( {target_col.get(obj)} );
                if (targetWs.size()==0) goto random_pick_done;
                targetW = targetWs.front();
                targetWs.pop_front();
            }
        }
        //safety against rounding error, should not reach here
        for (int i=0; i<(targetWs.size()+1);++i){
            selected_rows_col_val.update( {target_col.get_data().back()} );
        }
        random_pick_done:
        ;           
    }
    husky::LOG_I << "random at end " << dice() << std::endl;

    husky::lib::AggregatorFactory::sync();
    return selected_rows_col_val.get_value();
}

template <typename FeatureVec>
template <typename ObjT>
void Kmeans<FeatureVec>::fit(ObjList<ObjT>& obj_list, AttrList<ObjT,FeatureVec>& feature_col){

    if (!get_distance_sq) get_distance_sq = KmeansUtil::get_default_distance_sq_func( (FeatureVec*)0 );
    if (!get_distance_sq){ husky::LOG_I << "distance function lamba not given. Abort" << std::endl; return; }

    trained = true;

    int tid = husky::Context::get_global_tid();

    if (tid == 0) { husky::LOG_I <<"Kmeans init" ;}
    init(obj_list, feature_col);

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    std::vector<husky::lib::Aggregator<FeatureVec>> clusterCenterAggVec;
    std::vector<husky::lib::Aggregator<int>> clusterCountAggVec;
    husky::lib::Aggregator<int> switchClassCountAgg;
    for (int i = 0; i < k; i++) {
        clusterCenterAggVec.push_back(
            husky::lib::Aggregator<FeatureVec>(KmeansUtil::get_zero<FeatureVec>(dim), [&](FeatureVec& a, const FeatureVec& b) { a += b; },
                                               [&](FeatureVec& v) { v = std::move(KmeansUtil::get_zero<FeatureVec>(dim)); }));
        clusterCountAggVec.push_back(husky::lib::Aggregator<int>());
    }

    auto& class_col = obj_list.template create_attrlist<int>("KmeansClass");

    for (int it = 0; it < iter; ++it) {
        if (tid == 0) {
            husky::LOG_I << "iter " << it << " start";
        }

        // assign obj to centers
        list_execute(obj_list, {}, {&ac}, [&](ObjT& obj) {
            // get idx of nearest center
            FeatureVec& fea = feature_col.get(obj);
            int oldIdx = class_col.get(obj);
            int idx = getClass(fea);
            // assgin
            class_col.set(obj, idx);
            clusterCenterAggVec[idx].update(fea);
            clusterCountAggVec[idx].update(1);
            if (oldIdx != idx) switchClassCountAgg.update(1);
        });

        // update cluster center
        for (int i = 0; i < k; ++i) {
            int count = clusterCountAggVec[i].get_value();
            if (count != 0)
                clusterCenters[i] = clusterCenterAggVec[i].get_value() / count;

            clusterCenterAggVec[i].to_reset_each_iter();
            clusterCountAggVec[i].to_reset_each_iter();
        }

        if (switchClassCountAgg.get_value()==0) break;
        switchClassCountAgg.to_reset_each_iter();
    }
    return;
}

template <typename FeatureVec>
int Kmeans<FeatureVec>::getClass(const FeatureVec& v) {
    if (!trained) throw base::HuskyException("Kmeans not yet trained");
    double dist = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < k; ++i) {
        double d2 = get_distance_sq(v, clusterCenters[i]);
        if (d2 < dist) {
            dist = d2;
            idx = i;
        }
    }
    return idx;
}

/*
Usage:
see example/kmeans.cpp

*/
}  // namespace kmeans
}  // namespace ml
}  // namespace lib
}  // namespace husky
