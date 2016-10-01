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
#include "io/input/hdfs_line_inputformat.hpp"

void normalize(std::vector<double> &f, double rowsum) {
        for (auto & x : f) {
                    x /= rowsum;
                        }
}

double magnitude(std::vector<double> &a) {
        double sum = 0.0;
            for (auto & x : a) {
                        sum += pow(x, 2);
                            }
                return sqrt(sum);
}

double difference(std::vector<double> &a, std::vector<double> &b) {
        double sum = 0.0;
            int size = a.size();
                for (int i = 0; i < size; ++i) {
                            sum += pow(a[i] - b[i], 2);
                                }
                    return sum;
}

double similarity(std::vector<double> &a, std::vector<double> &b, double var) {
        double S_ab = exp(-(difference(a, b))/(2 * var));
            return S_ab;
}

class NDpoint {
   public:
    using KeyT = int;

    NDpoint() {}
    explicit NDpoint(const KeyT& w) : pointId(w) {}
    NDpoint(const KeyT& pId, std::vector<float> coords) {
        this->pointId = pId;
        this->coords  = std::move(coords);
    }
    virtual const KeyT& id() const { return pointId; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, NDpoint u) {
        stream << u.pointId<< u.coords;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, NDpoint& u) {
        stream >> u.pointId >> u.coords;
        return stream;
    }

    int pointId;
    std::vector<float> coords;
};

class DimStat {
   public:
    using KeyT = int;

    DimStat() {}
    explicit DimStat(const KeyT& w) : dimId(w) {}
    virtual const KeyT& id() const { return dimId; }

    int dimId;
    float mean;
    float var;
};

class PIObject {
   public:
    typedef int KeyT;
    int key;

    explicit PIObject(KeyT key) { this->key = key; }

    const int& id() const { return key; }
};

void pic() {
    husky::io::HDFSLineInputFormat infmt;
    infmt.set_input(husky::Context::get_param("input"));

    int dimension = std::stoi(husky::Context::get_param("dimension"));

    husky::base::log_msg( "input " + husky::Context::get_param("input"));
    husky::base::log_msg( "dim " + std::to_string(dimension));

    // 1. Create and globalize ndpoint objects
    husky::ObjList<NDpoint> ndpoint_list;
    auto parse_ndpoint = [&ndpoint_list](boost::string_ref& chunk) {
        husky::base::log_msg("pt1");
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator it = tok.begin();
        int id = stoi(*it++);
        it++;
        std::vector<float> coords;
        husky::base::log_msg("pt2");
        while (it != tok.end()) {
            coords.push_back(stod(*it++));
        }
        husky::base::log_msg( "data " + std::to_string(id) + " " +std::to_string(coords[0]) + " " + std::to_string(coords[1]));
        ndpoint_list.add_object(NDpoint(id, coords));
    };
    husky::base::log_msg( "part1a ok ");
    load(infmt, parse_ndpoint);
    husky::base::log_msg( "part1b ok ");
    globalize(ndpoint_list);
    husky::base::log_msg( "part1 ok ");
    
    // 2. calculate the variance used in similarity function
    husky::ObjList<DimStat> dimstat_list;
    auto& meanCh =
        husky::ChannelFactory::create_push_combined_channel<float, husky::SumCombiner<float>>(ndpoint_list, dimstat_list);
    auto& varCh =
        husky::ChannelFactory::create_push_combined_channel<float, husky::SumCombiner<float>>(ndpoint_list, dimstat_list);
    list_execute(ndpoint_list, [&meanCh, &varCh](NDpoint& p) {
        int idx=0;
        for (auto& aVal : p.coords) {
            meanCh.push(      aVal, idx);
            varCh .push( aVal*aVal, idx);
            idx++;
        }
    });

    size_t totDataPoints = ndpoint_list.get_size();
    husky::base::log_msg( "totPts" + totDataPoints);

    list_execute(dimstat_list, [&meanCh, &varCh, &totDataPoints](DimStat& d) {
        d.mean = meanCh.get(d)/totDataPoints;
        d.var  = sqrt(varCh.get(d)/totDataPoints - d.mean*d.mean); //Var(x) = sqrt(<x^2> - <x><x>)
        husky::base::log_msg( "stat " + std::to_string(d.id()) + "  " + std::to_string(d.mean) + " " + std::to_string(d.var));
    });

    husky::base::log_msg( "part2 ok ");

    //FIXME:: how to decide point pairing up to minimize traffic?
    //        should I store a local copy of dimstat_list?
    auto& pairCh =
        husky::ChannelFactory::create_push_channel<NDpoint>(ndpoint_list, ndpoint_list);

    list_execute(ndpoint_list, [&](NDpoint & p) {
        //FIXME:: the i for loop assumed the input data has index 0 to totDataPoints
        for (int i = 0; i < totDataPoints; ++i) {
            if (i == p.id()) { continue;} //don't pair up with oneself
            if      ((p.id()%2)==0 && (i%2)==1 ){ pairCh.push(p, i);}  //10 will push itself to 0 2 4 6 8
            else if ((p.id()%2)==1 && (i%2)==0 ){ pairCh.push(p, i);}  //10 will receive from 1 3 5 7 9
        }
    });

}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("dimension");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(pic);
        return 0;
    }
    return 1;
}
