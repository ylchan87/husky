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

namespace husky {
namespace lib {
namespace ml {

enum class KmeansOpts { kInitSimple, kInitKmeansPP };

template <typename ObjT, typename FeatureT>
class Kmeans {
   public:
    // interface
    void run();
    int getClass(const ObjT&);

    std::vector<FeatureT> clusterCenter;

    Kmeans() = default;

    // no init centers provided
    Kmeans(ObjList<ObjT>& obj_list, std::function<FeatureT(const ObjT&)> get_feature,
           std::function<double(const FeatureT&, const FeatureT&)> get_distance, const int k, const int iter,
           KmeansOpts kInitOption = KmeansOpts::kInitSimple)
        : obj_list(obj_list),
          get_feature(get_feature),
          get_distance(get_distance),
          k(k),
          iter(iter),
          kInitOption(kInitOption),
          clusterCenter() {}

    // init centers provided
    Kmeans(ObjList<ObjT>& obj_list, std::function<FeatureT(const ObjT&)> get_feature,
           std::function<double(const FeatureT&, const FeatureT&)> get_distance,
           std::vector<FeatureT> clusterCenter,  // initial center
           const int iter, KmeansOpts kInitOption = KmeansOpts::kInitSimple)
        : obj_list(obj_list),
          get_feature(get_feature),
          get_distance(get_distance),
          k(0),
          iter(iter),
          kInitOption(kInitOption),
          clusterCenter(clusterCenter) {}

   private:
    KmeansOpts kInitOption;

    ObjList<ObjT>& obj_list;
    std::function<FeatureT(const ObjT&)> get_feature;
    std::function<double(const FeatureT&, const FeatureT&)> get_distance;
    int k;
    int iter;
    int dim;

    std::vector<FeatureT>& init() {
        dim = get_feature(obj_list.get_data()[0]).get_feature_num();

        if (clusterCenter.size() == 0) {  // init_center not provided
            int worker_num = husky::Context::get_worker_info().get_num_workers();

            if (kInitOption == KmeansOpts::kInitSimple) {
                // an agg that concate vectors
                husky::lib::Aggregator<std::vector<FeatureT>> init_center_agg(
                    std::vector<FeatureT>(),
                    [](std::vector<FeatureT>& a, const std::vector<FeatureT>& b) {
                        a.insert(a.end(), b.begin(), b.end());
                    },
                    [](std::vector<FeatureT>& v) { v = std::move(std::vector<FeatureT>()); });

                // each worker choose a few centers and push them to agg
                int nSelect = k / worker_num + 1;
                std::vector<FeatureT> local_init_center;
                for (int i = 0; i < nSelect; i++) {
                    local_init_center.push_back(get_feature(obj_list.get_data()[i]));
                }
                init_center_agg.update(local_init_center);
                husky::lib::AggregatorFactory::sync();

                clusterCenter = init_center_agg.get_value();
                clusterCenter.resize(k);

            } else if (kInitOption == KmeansOpts::kInitKmeansPP) {
                husky::lib::Aggregator<FeatureT> nextCenter(FeatureT(dim, 0.),
                                                            [&](FeatureT& a, const FeatureT& b) { a += b; },
                                                            [&](FeatureT& v) { v = std::move(FeatureT(dim, 0.)); });

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
                int rand1 = int(dice() * worker_num);
                double rand2 = dice();
                if (tid == rand1) {
                    nextCenter.update(get_feature(localObjs[int(rand2 * localObjs.size())]));
                }
                husky::lib::AggregatorFactory::sync();
                clusterCenter.push_back(nextCenter.get_value());
                nextCenter.to_reset_each_iter();

                // pick remaining centers by prob weighted by squared distance to nearest chosen centers
                for (int i = 1; i < k; i++) {
                    // all datapoints are laid on a line, each span length equal to its weight (given by nearestD2)
                    // a point is randomly chosen on the line, the datapoint that this point falls on becomes next
                    // Kmeans center
                    double totw = 0.0;
                    for (int idx = 0; idx < localObjs.size(); ++idx) {
                        double d2 = get_distance(get_feature(localObjs[idx]), clusterCenter.back());
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
                    clusterCenter.push_back(nextCenter.get_value());
                    nextCenter.to_reset_each_iter();
                    for (auto atotw : totwVec) {
                        atotw.to_reset_each_iter();
                    }
                }  // end for loop over i= 1 to k
            }      // end if ( kInitOption == kInitKmeansPP)
        } else {   // init_center provided
            assert(clusterCenter[0].get_feature_num() == dim);
            k = clusterCenter.size();
        }
        return clusterCenter;
    }
};

template <typename ObjT, typename FeatureT>
void Kmeans<ObjT, FeatureT>::run() {
    int tid = husky::Context::get_global_tid();

    if (tid == 0) {
        husky::base::log_info("Kmeans init");
    }
    init();

    if (tid == 0) {
        husky::base::log_info("Kmeans run");
    }

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    std::vector<husky::lib::Aggregator<FeatureT>> clusterCenterAggVec;
    std::vector<husky::lib::Aggregator<int>> clusterCountAggVec;
    for (int i = 0; i < k; i++) {
        clusterCenterAggVec.push_back(
            husky::lib::Aggregator<FeatureT>(FeatureT(dim, 0.), [&](FeatureT& a, const FeatureT& b) { a += b; },
                                             [&](FeatureT& v) { v = std::move(FeatureT(dim, 0.)); }));
        clusterCountAggVec.push_back(husky::lib::Aggregator<int>());
    }

    for (int it = 0; it < iter; ++it) {
        // assign obj to centers
        if (tid == 0) {
            husky::base::log_info("iter " + std::to_string(it) + "start ");
        }
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
                clusterCenter[i] = clusterCenterAggVec[i].get_value() / count;

            clusterCenterAggVec[i].to_reset_each_iter();
            clusterCountAggVec[i].to_reset_each_iter();
        }
    }
    return;
}

template <typename ObjT, typename FeatureT>
int Kmeans<ObjT, FeatureT>::getClass(const ObjT& obj) {
    double dist = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < k; ++i) {
        double d = get_distance(get_feature(obj), clusterCenter[i]);
        if (d < dist) {
            dist = d;
            idx = i;
        }
    }
    return idx;
}

/*
Usage:

std::vector<featureVec> init_center(3, featureVec(dim,0.) );
init_center[0][0] = 0.;  init_center[0][1] = 1.;
init_center[1][0] =-1.;  init_center[1][1] =-1.;
init_center[2][0] = 1.;  init_center[2][1] = 1.;

auto kmeansOp = husky::lib::ml::Kmeans<Point, featureVec>(
    point_list,                                // obj_list
    [](const Point& p){ return p.features;},   // lambda to extraxt feature from obj
    euclidean_dist,                            // lambda to calculate distance between 2 features
    3,                                         // no. of centers to use, OR put init_center here
    maxIter,                                   // max iteration
    husky::lib::ml::KmeansOpts::kInitKmeansPP  // (optional arg) init method, kInitKmeansPP OR kInitSimple, irrelevant if init_center provided
);
kmeansOp.run();

*/
}  // namespace ml
}  // namespace lib
}  // namespace husky
