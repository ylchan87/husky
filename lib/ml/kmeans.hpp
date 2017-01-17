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
#include "lib/ml/feature_label.hpp"

namespace husky {
namespace lib {
namespace ml {

enum class KmeansOpts { kInitSimple, kInitKmeansPP };
namespace KmeansUtil{
    //primary template
    template<typename FeatureVec, typename ObjT>
    std::function<const FeatureVec(const ObjT&)> get_default_feature_extractor(FeatureVec*, ObjT*) {
        return nullptr;
    }

    template<typename FeatureVec>
    std::function<double(const FeatureVec&, const FeatureVec&)> get_default_distance_func(FeatureVec*) {
        return nullptr;
    }

    //specialize for lib::vector
    template<typename FeatureT, typename LabelT, bool is_sparse>
    using hObj = LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    template<typename FeatureT, bool is_sparse>
    using hVec = husky::lib::Vector<FeatureT, is_sparse>;

    template<typename FeatureT, typename LabelT, bool is_sparse>
    std::function<const hVec<FeatureT, is_sparse>&(const hObj<FeatureT, LabelT, is_sparse>&)>
    get_default_feature_extractor(hVec<FeatureT, is_sparse>*, hObj<FeatureT, LabelT, is_sparse>*) {
        return [](const hObj<FeatureT, LabelT, is_sparse>& o)->auto& { return o.x; };
    }

    template<typename FeatureT, bool is_sparse>
    std::function<double(const hVec<FeatureT, is_sparse>&, const hVec<FeatureT, is_sparse>&)>
    get_default_distance_func(hVec<FeatureT, is_sparse>*) {
        return [](const hVec<FeatureT, is_sparse>& v1, const hVec<FeatureT, is_sparse>& v2){ return v1.euclid_dist(v2);};
    }
}

template <typename FeatureVec>
class Kmeans {
   public:
    Kmeans(int k =1, int iter =1);

    template <typename ObjT>
    void fit(ObjList<ObjT>& obj_list, std::function<const FeatureVec(const ObjT&) > get_feature = nullptr);

    int getClass(const FeatureVec&);

    template <typename ObjT>
    int getClass(const ObjT&);

    Kmeans<FeatureVec>& set_k(int k){ this->k = k; return *this;}
    Kmeans<FeatureVec>& set_iter(int iter){ this->iter = iter; return *this;}
    Kmeans<FeatureVec>& set_feature_dim(int dim){ this->dim = dim; return *this;}
    Kmeans<FeatureVec>& set_init_opt(KmeansOpts opt){ this->kInitOption = opt; return *this;}

    Kmeans<FeatureVec>& set_distance_func(std::function<double(const FeatureVec& v1, const FeatureVec& v2)> f){
        this->get_distance = f;
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
    void init(ObjList<ObjT>& obj_list, std::function<const FeatureVec(const ObjT&) > get_feature);

    std::function<double(const FeatureVec& v1, const FeatureVec& v2)> get_distance;

};

template <typename FeatureVec>
Kmeans<FeatureVec>::Kmeans(int k, int iter) {
    this->k = k;
    this->iter = iter;
    this->kInitOption = KmeansOpts::kInitKmeansPP;
}

template <typename FeatureVec>
template <typename ObjT>
void Kmeans<FeatureVec>::init(ObjList<ObjT>& obj_list, std::function<const FeatureVec(const ObjT&) > get_feature) {
    if (clusterCenters.size() == 0) {  // init_center not provided
        int worker_num = husky::Context::get_worker_info().get_num_workers();

        // an agg that concate vectors
        husky::lib::Aggregator<std::vector<FeatureVec>> init_center_agg(
            std::vector<FeatureVec>(),
            [](std::vector<FeatureVec>& a, const std::vector<FeatureVec>& b) {
                a.insert(a.end(), b.begin(), b.end());
            },
            [](std::vector<FeatureVec>& v) { v = std::move(std::vector<FeatureVec>()); });

        if (kInitOption == KmeansOpts::kInitSimple) {

            // each worker choose a few centers and push them to agg
            int nSelect = std::min(k, (int)obj_list.get_data().size());
            std::vector<FeatureVec> local_init_center;
            for (int i = 0; i < nSelect; i++) {
                husky::LOG_I << "i" << i << std::endl;
                local_init_center.push_back(get_feature(obj_list.get_data()[i]));
            }
            init_center_agg.update(local_init_center);
            husky::lib::AggregatorFactory::sync();

            clusterCenters = init_center_agg.get_value();
            clusterCenters.resize(k);

        } else if (kInitOption == KmeansOpts::kInitKmeansPP) {
            husky::lib::Aggregator<FeatureVec> nextCenter(FeatureVec(dim, 0.),
                                                          [&](FeatureVec& a, const FeatureVec& b) { a += b; },
                                                          [&](FeatureVec& v) { v = std::move(FeatureVec(dim, 0.)); });

            std::vector<husky::lib::Aggregator<double>> totwVec;
            for (int i = 0; i < worker_num; i++) {
                totwVec.push_back(husky::lib::Aggregator<double>());
            }

            int tid = husky::Context::get_global_tid();
            auto& localObjs = obj_list.get_data();
            std::vector<double> nearestD2(localObjs.size(), DBL_MAX);

            std::default_random_engine generator;
            generator.seed(1234);
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            auto dice = std::bind(distribution, generator);

            // randomly pick the first center
            double rand1 = dice();
            std::vector<FeatureVec> local_init_center;
            if (localObjs.size()>0) {
                local_init_center.push_back(get_feature(localObjs[int(rand1 * localObjs.size())]));
            }
            init_center_agg.update(local_init_center);
            husky::lib::AggregatorFactory::sync();
            clusterCenters = init_center_agg.get_value();
            clusterCenters.resize(1);

            // pick remaining centers by prob weighted by squared distance to nearest chosen centers
            for (int i = 1; i < k; i++) {
                // all datapoints are laid on a line, each span length equal to its weight (given by nearestD2)
                // a point is randomly chosen on the line, the datapoint that this point falls on becomes next
                // Kmeans center
                double totw = 0.0;
                for (int idx = 0; idx < localObjs.size(); ++idx) {
                    double d2 = get_distance(get_feature(localObjs[idx]), clusterCenters.back());
                    d2 *= d2;
                    if (d2 < nearestD2[idx]) {
                        nearestD2[idx] = d2;
                    }
                    totw += nearestD2[idx];
                }
                totwVec[tid].update(totw);
                husky::lib::AggregatorFactory::sync();

                totw = 0.0;
                for (auto atotw : totwVec) {
                    totw += atotw.get_value();
                }
                double targetSumW = totw * dice();

                for (int iworker = 0; iworker < worker_num; ++iworker) {
                    double thisWorkerSumW = totwVec[iworker].get_value();
                    if (thisWorkerSumW > targetSumW) {
                        // the random point chosen lie within the segment this worker has
                        if (tid == iworker) {
                            bool targetFound = false;
                            for (int idx = 0; idx < localObjs.size(); ++idx) {
                                if (nearestD2[idx] > targetSumW) {
                                    nextCenter.update(get_feature(localObjs[idx]));
                                    targetFound = true;
                                    break;
                                } else {
                                    targetSumW -= nearestD2[idx];
                                }
                            }
                            // safety against numerical error
                            if (!targetFound) {
                                nextCenter.update(get_feature(localObjs.back()));
                            }
                        }
                        break;
                    } else {
                        targetSumW -= thisWorkerSumW;
                    }
                }
                husky::lib::AggregatorFactory::sync();
                clusterCenters.push_back(nextCenter.get_value());
                nextCenter.to_reset_each_iter();
                for (auto atotw : totwVec) {
                    atotw.to_reset_each_iter();
                }
            }  // end for loop over i= 1 to k
        }      // end if ( kInitOption == kInitKmeansPP)
    } else {   // init_center provided
        if(clusterCenters[0].get_feature_num() != dim){
            throw husky::base::HuskyException("User provided centers feature dim differs from Kmeans expectation");
        }
        k = clusterCenters.size();
    }
}

template <typename FeatureVec>
template <typename ObjT>
void Kmeans<FeatureVec>::fit( ObjList<ObjT>& obj_list, std::function<const FeatureVec(const ObjT&) > get_feature) {

    if (!get_feature) get_feature = KmeansUtil::get_default_feature_extractor( (FeatureVec*)0, (ObjT*)0 );
    if (!get_distance) get_distance = KmeansUtil::get_default_distance_func( (FeatureVec*)0 );

    if (!get_feature ){ husky::LOG_I << "feature extractor lamba not given. Abort" << std::endl; return; }
    if (!get_distance){ husky::LOG_I << "distance function lamba not given. Abort" << std::endl; return; }

    trained = true;

    int tid = husky::Context::get_global_tid();

    if (tid == 0) { husky::LOG_I <<"Kmeans init" ;}
    init(obj_list, get_feature);

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    std::vector<husky::lib::Aggregator<FeatureVec>> clusterCenterAggVec;
    std::vector<husky::lib::Aggregator<int>> clusterCountAggVec;
    for (int i = 0; i < k; i++) {
        clusterCenterAggVec.push_back(
            husky::lib::Aggregator<FeatureVec>(FeatureVec(dim, 0.), [&](FeatureVec& a, const FeatureVec& b) { a += b; },
                                               [&](FeatureVec& v) { v = std::move(FeatureVec(dim, 0.)); }));
        clusterCountAggVec.push_back(husky::lib::Aggregator<int>());
    }

    for (int it = 0; it < iter; ++it) {
        if (tid == 0) {
            husky::LOG_I << "iter " << it << " start";
        }

        // assign obj to centers
        list_execute(obj_list, {}, {&ac}, [&](ObjT& obj) {
            // get idx of nearest center
            int idx = getClass(get_feature(obj));
            // assgin
            clusterCenterAggVec[idx].update(get_feature(obj));
            clusterCountAggVec[idx].update(1);
        });

        // update cluster center
        for (int i = 0; i < k; ++i) {
            int count = clusterCountAggVec[i].get_value();
            if (count != 0)
                clusterCenters[i] = clusterCenterAggVec[i].get_value() / count;

            clusterCenterAggVec[i].to_reset_each_iter();
            clusterCountAggVec[i].to_reset_each_iter();
        }
    }
    return;
}

template <typename FeatureVec>
int Kmeans<FeatureVec>::getClass(const FeatureVec& v) {
    if (!trained) throw base::HuskyException("Kmeans not yet trained");
    double dist = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < k; ++i) {
        double d = get_distance(v, clusterCenters[i]);
        if (d < dist) {
            dist = d;
            idx = i;
        }
    }
    return idx;
}

template <typename FeatureVec>
template <typename ObjT>
int Kmeans<FeatureVec>::getClass(const ObjT& o) {
    if (!trained) throw base::HuskyException("Kmeans not yet trained");
    auto get_feature = KmeansUtil::get_default_feature_extractor( (FeatureVec*)0, (ObjT*)0 );
    if (!get_feature ) throw base::HuskyException("no default feature extractor for given object");
    return this->getClass( get_feature(o) );
}

/*
Usage:
see example/kmeans.cpp

*/
}  // namespace ml
}  // namespace lib
}  // namespace husky
