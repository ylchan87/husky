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

void testKmeans() {

    int maxIter = std::stoi(husky::Context::get_param("iter"));

    // 1. read data
    auto& point_list = husky::ObjListStore::create_objlist<LabeledPointHObj>("point_list");
    int dim = husky::lib::ml::load_data(husky::Context::get_param("input"), point_list, husky::lib::ml::kLIBSVMFormat);
    husky::base::log_info("local point_list size " + std::to_string(point_list.get_data().size()));

    // 2. train
    auto kmeansOp = husky::lib::ml::Kmeans<double, int, false>(3,maxIter); //template arg: feature is a double non-sparse vec, label is int (not used)
    kmeansOp.setInitOpt(husky::lib::ml::KmeansOpts::kInitKmeansPP).setFeatureDim(dim);
    kmeansOp.fit(point_list);

    // 3. output k means centers
    if (husky::Context::get_global_tid() == 0) {
        auto clusterCenter = kmeansOp.getCenters();
        std::ostringstream oss;
        oss << "kmeans: " << std::endl;
        for (int i = 0; i < clusterCenter.size(); ++i) {
            oss << i << " ";
            for (auto aVal : clusterCenter[i]) { oss << aVal << " ";}
            oss << std::endl;
        }
        husky::base::log_info(oss.str());
    }

    // 4. classify some points
    if (point_list.get_data().size()>0){
        husky::base::log_info("point class " + std::to_string(kmeansOp.getClass( point_list.get_data()[0])));
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
