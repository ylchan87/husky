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

#include "boost/tokenizer.hpp"

#include "core/engine.hpp"
#include "io/hdfs_manager.hpp"
#include "io/input/line_inputformat.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/table_loader.hpp"

void testTableLoader() {
    
    ////--------------------------------------------------------------------------------------
    ////Reading a input file with lines like:
    ////0 -2.4903068672 3.13023508816
    ////--------------------------------------------------------------------------------------
    //husky::LOG_I << "test 1" << std::endl;
    //auto loader1 = husky::lib::ml::tableloader::get_instance<int, double, double>();
    //loader1.set_col_names( { "idx", "x", "y"});
    //auto& table1 = loader1.load(husky::Context::get_param("input"));

    //auto& col_x = table1.get_attrlist<double>("x");
    //if (col_x.size()>0) husky::LOG_I << "x999" << col_x[999] << std::endl;

    //--------------------------------------------------------------------------------------
    //Reading a input file with lines like:
    //0 -2.4903068672 3.13023508816
    //--------------------------------------------------------------------------------------
    husky::LOG_I << "test 2" << std::endl;
    auto loader2 = husky::lib::ml::tableloader::get_instance<int, husky::lib::VectorXd>();
    loader2.set_col_names( { "idx", "xy"});
    auto& table2 = loader2.load("hdfs:///user/ylchan/testPICdata.txt");

    auto& col_xy = table2.get_attrlist<husky::lib::VectorXd>("xy");
    if (col_xy.size()>0) husky::LOG_I << "xy999" << col_xy[999][0] << std::endl;

    //--------------------------------------------------------------------------------------
    //Reading a input file with lines like (i.e. LIBSVM format):
    //0 0:-5.7362599581E-03 1:1.2793435994E-03 2:6.9297686188E-03
    //--------------------------------------------------------------------------------------
    husky::LOG_I << "test 3" << std::endl;
    auto loader3 = husky::lib::ml::tableloader::get_libsvm_instance();
    loader3.set_col_names( { "idx", "featureVec"});
    auto& table3 = loader3.load("hdfs:///user/ylchan/obsDataSparse_10D.txt");

    auto& col_feat = table3.get_attrlist<husky::lib::SparseVectorXd>("featureVec");
    if (col_feat.size()>0) husky::LOG_I << "feat" << col_feat[9].coeffRef(5) << std::endl;
    if (col_feat.size()>0) husky::LOG_I << "feat" << col_feat[9].size() << std::endl;

    //--------------------------------------------------------------------------------------
    //Reading a input file with lines like (i.e. csv format):
    //0, red apple, 10, 2.3
    //--------------------------------------------------------------------------------------
    husky::LOG_I << "test 4" << std::endl;
    auto loader4 = husky::lib::ml::tableloader::get_csv_instance<int,std::string,int,double>();
    loader4.set_col_names( { "idx", "itemname", "count", "price"});
    auto& table4 = loader4.load("hdfs:///user/ylchan/testCSVdata.txt");

    auto& col_itemname = table4.get_attrlist<std::string>("itemname");
    auto& col_price    = table4.get_attrlist<double>("price");
    if (col_itemname.size()>0) husky::LOG_I << col_itemname[0] << " " << col_price[0] << std::endl;
    //--------------------------------------------------------------------------------------
    husky::LOG_I << "test done" << std::endl;

}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");  // path to input file eg. hdfs:///user/ylchan/testKmeansData_6000_K3_LIBSVMfmt.txt
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(testTableLoader);
        return 0;
    }
    return 1;
}
