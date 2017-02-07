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

//#define HUSKY_DEBUG_MODE 1

#include <random>
#include <string>
#include <vector>

#include "lib/ml/kmeans.hpp"
#include "lib/ml/table_loader.hpp"
#include "lib/vector.hpp"

#include "core/engine.hpp"

void testKmeans() {
    int maxIter = std::stoi(husky::Context::get_param("iter"));

    // 1. read data
    auto loader = husky::lib::ml::tableloader::get_libsvm_instance();
    loader.set_col_names({"idx", "feature"});
    auto& table = loader.load(husky::Context::get_param("input"));
    auto& featureCol = table.get_attrlist<husky::lib::SparseVectorXd>("feature");

    // 2. train
    auto kmeansInst = husky::lib::ml::kmeans::Kmeans<husky::lib::SparseVectorXd>(3, maxIter);
    kmeansInst.set_init_opt(husky::lib::ml::kmeans::KmeansOpts::kInitKmeansBarBar);
    kmeansInst.set_center_sample_n_iter(3).set_center_sample_per_iter(3);
    kmeansInst.fit(table, featureCol);  // this will add a new AttrList "KmeansClass" to the table

    // 3. output k means centers
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "Trained centers:";
        std::vector<husky::lib::SparseVectorXd> clusterCenters = kmeansInst.get_centers();
        for (auto aCenter : clusterCenters) {
            husky::LOG_I << aCenter;
        }
    }

    // 4. get class of training data
    auto& classCol = table.get_attrlist<int>("KmeansClass");
    if (classCol.get_data().size() > 0) {
        husky::LOG_I << "class for a row in training data: " << classCol.get_data()[0];
    }

    // 5. classify some new points
    if (featureCol.get_data().size() > 0) {
        int dim = featureCol.get_data()[0].size();
        husky::LOG_I << "class for new point: " << kmeansInst.getClass(husky::lib::SparseVectorXd(dim));
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
