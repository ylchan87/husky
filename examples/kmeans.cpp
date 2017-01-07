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

#include <random>
#include <string>
#include <vector>

#include "lib/ml/data_loader.hpp"
#include "lib/ml/kmeans.hpp"
#include "lib/vector.hpp"

#include "core/engine.hpp"

// kmeans expect featureVec to support operator/ and operator+=, and method get_feature_num()
typedef husky::lib::DenseVector<double> featureVec;
using LabeledPointHObj = husky::lib::ml::LabeledPointHObj<double, double, false>;

double euclidean_dist(const featureVec& v1, const featureVec& v2) { return v1.euclid_dist(v2); }

void testKmeans() {

    int maxIter = std::stoi(husky::Context::get_param("iter"));

    // 1. read data
    auto& point_list = husky::ObjListStore::create_objlist<LabeledPointHObj>("point_list");
    int dim = husky::lib::ml::load_data(husky::Context::get_param("input"), point_list, husky::lib::ml::kLIBSVMFormat);

    husky::base::log_info("local point_list size " + std::to_string(point_list.get_data().size()));

    // 2. train
    std::vector<featureVec> init_center(3, featureVec(2, 0.));
    init_center[0][0] = 0.;
    init_center[0][1] = 1.;
    init_center[1][0] = -1.;
    init_center[1][1] = -1.;
    init_center[2][0] = 1.;
    init_center[2][1] = 1.;

    auto kmeansOp = husky::lib::ml::Kmeans<LabeledPointHObj, featureVec>(3,maxIter);
    kmeansOp.setFeatureExtractor( [](const LabeledPointHObj& p){ return p.x;}); // lambda to extraxt feature from obj
    kmeansOp.setDistanceFunc(euclidean_dist);                                   // lambda to calculate distance between 2 features
    kmeansOp.setInitOpt(husky::lib::ml::KmeansOpts::kInitKmeansPP).setFeatureDim(dim);

    kmeansOp.fit(point_list);

    // 3. output
    if (husky::Context::get_global_tid() == 0) {
        auto clusterCenter = kmeansOp.getCenters();
        std::ostringstream oss;
        oss << std::scientific;
        oss.precision(17);
        oss << "kmeans: " << std::endl;
        for (int i = 0; i < clusterCenter.size(); ++i) {
            oss << i << " ";
            for (auto aVal : clusterCenter[i]) {
                oss << aVal << " ";
            }
            oss << std::endl;
        }
        husky::base::log_info(oss.str());
    }
}

int main(int argc, char** argv) {
    // Input:
    // A txt with each line containing 1 row of the data as
    // rowID rowElem1 rowElem2 rowElem3 ...rowElemN
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");  // path to input file eg. hdfs:///user/ylchan/AffMat_T2/merge
    args.push_back("iter");   // max no. of iteration to do
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(testKmeans);
        return 0;
    }
    return 1;
}
