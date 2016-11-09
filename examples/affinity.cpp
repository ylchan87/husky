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

#include "lib/vector.hpp"

typedef husky::lib::DenseVector<float> ndvec;

class NDpoint {
    // N-dimension point modelling the input data
   public:
    using KeyT = int;

    NDpoint() {}
    explicit NDpoint(const KeyT& w) : pointId(w) {}
    NDpoint(const KeyT& pId, ndvec coords) {
        this->pointId = pId;
        this->coords = std::move(coords);
    }
    virtual const KeyT& id() const { return pointId; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, NDpoint u) {
        stream << u.pointId << u.coords;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, NDpoint& u) {
        stream >> u.pointId >> u.coords;
        return stream;
    }

    int pointId;
    ndvec coords;
};

class MatElem {
    // Matrix elemet for the affinity matrix
   public:
    using KeyT = uint;

    MatElem() {}
    explicit MatElem(const KeyT& p) : pos(p) {}
    virtual const KeyT& id() const { return pos; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, MatElem u) {
        stream << u.pos << u.val;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, MatElem& u) {
        stream >> u.pos >> u.val;
        return stream;
    }

    KeyT pos;
    float val;
};

class MatRow {
    // Dummy obj to collect Matrix elemet for the affinity matrix
   public:
    using KeyT = uint;

    MatRow() {}
    explicit MatRow(const KeyT& p) : pos(p) {}
    virtual const KeyT& id() const { return pos; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, MatRow u) {
        stream << u.pos;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, MatRow& u) {
        stream >> u.pos;
        return stream;
    }

    KeyT pos;
};

void affinity() {
    husky::io::LineInputFormat infmt;
    infmt.set_input(husky::Context::get_param("input"));

    int dimension = std::stoi(husky::Context::get_param("dimension"));

    // for calculating mean(x_i  ) for each dimension i in input data
    husky::lib::Aggregator<ndvec> dataxAgg(ndvec(dimension, 0.), [](ndvec& a, const ndvec& b) { a += b; },
                                           [&](ndvec& v) { v = std::move(ndvec(dimension, 0.)); });

    // for calculating mean(x_i^2) for each dimension i in input data
    husky::lib::Aggregator<ndvec> dataxxAgg(ndvec(dimension, 0.), [](ndvec& a, const ndvec& b) { a += b; },
                                            [&](ndvec& v) { v = std::move(ndvec(dimension, 0.)); });

    husky::lib::Aggregator<int> dataCountAgg;  // total no. of rows in input
    auto& ac = husky::lib::AggregatorFactory::get_channel();

    // 1. Create and globalize ndpoint objects
    auto& ndpoint_list = husky::ObjListFactory::create_objlist<NDpoint>();
    ndvec coordsSq(dimension);
    auto parse_ndpoint = [&](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator it = tok.begin();
        int id = stoi(*it++);
        ndvec coords(dimension);
        for (int i = 0; i < dimension; i++) {
            coords[i] = stod(*it++);
            coordsSq[i] = coords[i] * coords[i];
        }
        dataCountAgg.update(1);
        dataxAgg.update(coords);
        dataxxAgg.update(coordsSq);
        ndpoint_list.add_object(NDpoint(id, coords));
    };
    load(infmt, {&ac}, parse_ndpoint);
    globalize(ndpoint_list);
    husky::base::log_msg("No. of rows in input " + std::to_string(dataCountAgg.get_value()));

    // 2. calculate the variance used in similarity function
    size_t totDataPoints = dataCountAgg.get_value();
    ndvec dataMean = dataxAgg.get_value() / totDataPoints;
    ndvec dataVar = dataxxAgg.get_value() / totDataPoints;
    // Var(x) = <x^2> - <x><x>
    for (int i = 0; i < dimension; ++i) {
        dataVar[i] -= dataMean[i] * dataMean[i];
        husky::base::log_msg("var " + std::to_string(i) + " " + std::to_string(dataVar[i]));
    }

    // 3. calculate the affinity matrix
    husky::ObjList<MatElem> affMatElem_list;
    globalize(affMatElem_list);
    auto& pairCh = husky::ChannelFactory::create_push_channel<NDpoint>(ndpoint_list, affMatElem_list);

    // push 2 data points to each matrix element above the diagonal
    list_execute(ndpoint_list, [&](NDpoint& p) {
        // the pid-th col
        for (uint i = 0; i < p.id(); ++i) {
            pairCh.push(p, i * totDataPoints + p.id());
        }
        // the pid-th row
        for (uint i = p.id() + 1; i < totDataPoints; ++i) {
            pairCh.push(p, p.id() * totDataPoints + i);
        }
    });

    husky::base::log_msg("affMat size " + std::to_string(affMatElem_list.get_size()));

    husky::ObjList<MatRow> affMatRow_list;
    auto& elemToRowCh = husky::ChannelFactory::create_push_channel<MatElem>(affMatElem_list, affMatRow_list);

    // group matrix elements to rows
    list_execute(affMatElem_list, [&](MatElem& e) {
        auto pts = pairCh.get(e);
        float nSigmaSq = 0;
        for (uint i = 0; i < dimension; i++) {
            nSigmaSq += pow(pts[0].coords[i] - pts[1].coords[i], 2) / (2. * dataVar[i]);
        }
        e.val = exp(-nSigmaSq);
        int col = e.id() % totDataPoints;
        int row = e.id() / totDataPoints;
        // As affinity matrix is symmetric, each matrix element above the diagonal "belongs" to 2 rows
        elemToRowCh.push(e, col);
        elemToRowCh.push(e, row);
    });

    // 4. print output
    list_execute(affMatRow_list, [&](MatRow& r) {
        auto elems = elemToRowCh.get(r);
        std::vector<float> rowVec(totDataPoints);
        for (auto aElem : elems) {
            int elemCol = aElem.id() % totDataPoints;
            int elemRow = aElem.id() / totDataPoints;
            if (elemCol > r.id())
                rowVec[elemCol] = aElem.val;
            else
                rowVec[elemRow] = aElem.val;
        }
        std::string log;
        log = std::to_string(r.id());
        for (auto aElem : rowVec) {
            log += "  " + std::to_string(aElem);
        }
        log += "\n";
        husky::io::HDFS::Write("master", "9000", log, husky::Context::get_param("outDir"),
                               husky::Context::get_global_tid());
    });
}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("outDir");
    args.push_back("dimension");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(affinity);
        return 0;
    }
    return 1;
}
