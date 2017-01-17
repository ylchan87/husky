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

using LabeledPointHObj = husky::lib::ml::LabeledPointHObj<double, int, false>;

void print_centers( std::vector<LabeledPointHObj::FeatureV>& clusterCenters){ 
    std::ostringstream oss;
    oss << "kmeans: " << std::endl;
    for (int i = 0; i < clusterCenters.size(); ++i) {
        oss << i << " ";
        for (auto aVal : clusterCenters[i]) { oss << aVal << " ";}
        oss << std::endl;
    }
    husky::LOG_I << oss.str();
}

void testKmeans() {

    int maxIter = std::stoi(husky::Context::get_param("iter"));

    // 1. read data
    auto& point_list = husky::ObjListStore::create_objlist<LabeledPointHObj>("point_list");
    int dim = husky::lib::ml::load_data(husky::Context::get_param("input"), point_list, husky::lib::ml::kLIBSVMFormat);
    husky::LOG_I << "local point_list size " << point_list.get_data().size();

    // 2. train
    auto kmeansInst = husky::lib::ml::Kmeans<LabeledPointHObj::FeatureV>(3,maxIter);
    kmeansInst.set_init_opt(husky::lib::ml::KmeansOpts::kInitKmeansPP).set_feature_dim(dim);
    kmeansInst.fit(point_list);

    // 3. output k means centers
    if (husky::Context::get_global_tid() == 0) {
        std::vector<LabeledPointHObj::FeatureV> clusterCenters = kmeansInst.get_centers();
        print_centers(clusterCenters);
    }

    // 4. classify some points
    if (point_list.get_data().size()>0){
        husky::LOG_I << "point class " << kmeansInst.getClass(point_list.get_data()[0]);
    }

    //Fancy usage, do Kmeans on the 1st element in feature vector
    // A. Create a Kmeans instance as in basic usage
    auto kmeansInst2 = husky::lib::ml::Kmeans<LabeledPointHObj::FeatureV>(2,maxIter);
    kmeansInst2.set_init_opt(husky::lib::ml::KmeansOpts::kInitKmeansPP).set_feature_dim(1);

    // B. provide a custom def of distance that only counts the 1st element
    kmeansInst2.set_distance_func( [](LabeledPointHObj::FeatureV v1, LabeledPointHObj::FeatureV v2){ return abs(v1[0]-v2[0]);} );

    // C. provide a custom feature extractor that get the 1st feature element from the husky object
    kmeansInst2.fit<LabeledPointHObj>(point_list, [](const LabeledPointHObj& o){return LabeledPointHObj::FeatureV(1,o.x[0]);} );

    if (husky::Context::get_global_tid() == 0) {
        std::vector<LabeledPointHObj::FeatureV> clusterCenters = kmeansInst2.get_centers();
        print_centers(clusterCenters);
    }
}

int main(int argc, char** argv) {
    // Input:
    // A txt with LIBSVM format
    // label idx1:val1 idx2:val2 ...
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");  // path to input file eg. hdfs:///user/ylchan/testKmeansData_6000_K3_LIBSVMfmt.txt
    args.push_back("iter");   // max no. of iteration to do
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(testKmeans);
        return 0;
    }
    return 1;
}
