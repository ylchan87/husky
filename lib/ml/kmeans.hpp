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
#include "lib/aggregator_factory.hpp"
#include "lib/ml/feature_label.hpp"

namespace husky {
namespace lib {
namespace ml {

enum class KmeansOpts { kInitSimple, kInitKmeansPP };

template <typename FeatureT, typename LabelT, bool is_sparse>
class Kmeans {
   using ObjT = LabeledPointHObj<FeatureT, LabelT, is_sparse>;
   using FeatureVec = husky::lib::Vector<FeatureT, is_sparse>;
   public:
    // interface
    Kmeans(int k =1, int iter =1);
    void fit(ObjList<ObjT>& obj_list);
    int getClass(const ObjT&);

    Kmeans<FeatureT, LabelT, is_sparse>& setK(int k){ this->k = k; return *this;}
    Kmeans<FeatureT, LabelT, is_sparse>& setIter(int iter){ this->iter = iter; return *this;}
    Kmeans<FeatureT, LabelT, is_sparse>& setFeatureDim(int dim){ this->dim = dim; return *this;}
    Kmeans<FeatureT, LabelT, is_sparse>& setInitOpt(KmeansOpts opt){ this->kInitOption = opt; return *this;}

    Kmeans<FeatureT, LabelT, is_sparse>& setCenters(std::vector<FeatureVec> clusterCenters){
      this->clusterCenters = clusterCenters;
      return *this;
    }

    const std::vector<FeatureVec>& getCenters(){
      assert(trained);
      return this->clusterCenters;
    }

   private:
    bool trained;

    int k;
    int iter;
    int dim;    
    KmeansOpts kInitOption;

    std::vector<FeatureVec> clusterCenters;

    void init(ObjList<ObjT>& obj_list);
    const FeatureVec& get_feature(const ObjT& o);
    double get_distance(const FeatureVec& v1, const FeatureVec& v2);


};

template <typename FeatureT, typename LabelT, bool is_sparse>
Kmeans<FeatureT, LabelT, is_sparse>::Kmeans(int k, int iter) {
    this->k = k;
    this->iter = iter;
    this->kInitOption = KmeansOpts::kInitKmeansPP;
}

template <typename FeatureT, typename LabelT, bool is_sparse>
const husky::lib::Vector<FeatureT, is_sparse>& Kmeans<FeatureT, LabelT, is_sparse>::get_feature(const ObjT& o) {
    return o.x;
}

template <typename FeatureT, typename LabelT, bool is_sparse>
double Kmeans<FeatureT, LabelT, is_sparse>::get_distance(const FeatureVec& v1, const FeatureVec& v2){
    return v1.euclid_dist(v2);
}

template <typename FeatureT, typename LabelT, bool is_sparse>
void Kmeans<FeatureT, LabelT, is_sparse>::init(ObjList<ObjT>& obj_list) {
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
        assert(clusterCenters[0].get_feature_num() == dim);
        k = clusterCenters.size();
    }
}

template <typename FeatureT, typename LabelT, bool is_sparse>
void Kmeans<FeatureT, LabelT, is_sparse>::fit( ObjList<ObjT>& obj_list) {
    trained = true;

    int tid = husky::Context::get_global_tid();

    if (tid == 0) { husky::LOG_I <<"Kmeans init" ;}
    init(obj_list);

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
            int idx = getClass(obj);
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

template <typename FeatureT, typename LabelT, bool is_sparse>
int Kmeans<FeatureT, LabelT, is_sparse>::getClass(const ObjT& obj) {
    assert(trained);
    double dist = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < k; ++i) {
        double d = get_distance(get_feature(obj), clusterCenters[i]);
        if (d < dist) {
            dist = d;
            idx = i;
        }
    }
    return idx;
}

/*
Usage:
see example/kmeans.cpp

*/
}  // namespace ml
}  // namespace lib
}  // namespace husky
