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

#include "kmeans.hpp"

#include "boost/tokenizer.hpp"

#include "core/engine.hpp"
#include "io/hdfs_manager.hpp"
#include "io/input/line_inputformat.hpp"
#include "lib/aggregator_factory.hpp"

typedef std::vector<double> featureVec;

// kmeans expect featureVec to support operator/ and operator+=
featureVec operator/ (const featureVec& v1, const int d) {
    assert(d != 0);
    featureVec ret(v1.size());
    for (int i = 0; i < v1.size(); ++ i)
        ret[i] = v1[i]/d;
    return ret;
}

featureVec& operator+=(featureVec& v1, const featureVec& v2) {
    if (v1.size()==0) { v1=v2; return v1;}
    if (v2.size()==0) return v1;
    for (int i = 0; i < v1.size(); ++ i)
        v1[i] += v2[i];
    return v1;
}

featureVec operator+(const featureVec& v1, const featureVec& v2) {
    if (v1.size()==0) return featureVec(v2);
    if (v2.size()==0) return v1;
    featureVec v3( v1.size());
    for (int i = 0; i < v1.size(); ++ i)
        v3[i] = v1[i] + v2[i];
    return v3;
}

// Euclidean distance
double euclidean_dist(const featureVec& v1, const featureVec& v2){
    double ret = 0;
    for (int i = 0; i < v1.size(); ++ i) {
        ret += pow(v1[i]-v2[i], 2);
    }
    return sqrt(ret);
};


class Point {
   public:
    using KeyT = int;

    Point() {}
    explicit Point(const KeyT& k) : key(k){}
    Point(const KeyT& k, featureVec& fV) {
        this->key = k;
        this->features = std::move(fV);
    }
    virtual const KeyT& id() const { return key; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, Point p) {
        stream << p.key << p.features;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, Point& p) {
        stream >> p.key >> p.features;
        return stream;
    }

    KeyT key;
    featureVec features;

};

void testKmeans() {
    husky::io::LineInputFormat infmt;
    infmt.set_input(husky::Context::get_param("input"));

    int maxIter = std::stoi(husky::Context::get_param("iter"));

    husky::lib::Aggregator<int> dataCountAgg;
    auto& ac = husky::lib::AggregatorFactory::get_channel();

    // 1. Create and globalize point objects
    auto& point_list = husky::ObjListStore::create_objlist<Point>();
    auto parse_row = [&](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator it = tok.begin();
        int id = stoi(*it++);
        featureVec fV;
        while (it != tok.end()) {
            float tmp = stod(*it++);
            fV.push_back(tmp);
        }
        fV.shrink_to_fit();

        dataCountAgg.update(1);
        point_list.add_object(Point(id, fV));
    };
    load(infmt, {&ac}, parse_row);
    husky::globalize(point_list);

    int totDataPoints = dataCountAgg.get_value();
    husky::base::log_info("point_list size " + std::to_string(totDataPoints));

    // 2. train
    std::vector<featureVec> init_center = {{0,1}, {-1,-1}, {1,1}};
    auto kmeansOp = husky::Kmeans<Point, featureVec>(
        point_list, 
        [](const Point& p){ return p.features;},
        euclidean_dist,
        //init_center,
        3,
        maxIter,
        //husky::KmeansOpts::kInitSimple
        husky::KmeansOpts::kInitKmeansPP
    );
    kmeansOp.run();


    // 3. output
    if (husky::Context::get_global_tid()==0){
      auto clusterCenter  = kmeansOp.clusterCenter;
      std::ostringstream oss;
      oss << std::scientific; oss.precision(17);
      oss << "kmeans: " << std::endl;
      for (int i=0;i<clusterCenter.size();++i){ 
        oss << i << " ";
        for (auto aVal: clusterCenter[i]){ oss << aVal << " "; }
        oss << std::endl;
      }
      husky::base::log_info( oss.str() );
    }
}

int main(int argc, char** argv) {
    // Input:
    // A txt with each line containing 1 row of the data as
    // rowID rowElem1 rowElem2 rowElem3 ...rowElemN
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");        // path to input file eg. hdfs:///user/ylchan/AffMat_T2/merge
    args.push_back("iter");         // max no. of iteration to do
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(testKmeans);
        return 0;
    }
    return 1;
}
