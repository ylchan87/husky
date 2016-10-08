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
#include "lib/aggregator_factory.hpp"


class NDpoint {
   //N-dimension point modelling the input data
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

class MatElem {
   //Matrix elemet for the affinity matrix
   public:
    //using KeyT = std::pair<uint, uint>; // std::hash don't like std::pair
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

class VElem {
   //Element for the classification vector V
   public:
    //using KeyT = std::pair<uint, uint>; // std::hash don't like std::pair
    using KeyT = int;

    VElem() {}
    explicit VElem(const KeyT& p) : pos(p), val(0.), valp(0.), valpp(0.) {}
    virtual const KeyT& id() const { return pos; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, VElem u) {
        stream << u.pos << u.val << u.valp << u.valpp;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, VElem& u) {
        stream >> u.pos >> u.val >> u.valp >> u.valpp;
        return stream;
    }

    KeyT pos;
    float val;
    float valp;  //val at previous iteration
    float valpp; //val at previousX2 iteration

    void setVal(float newval){
        valpp = valp;
        valp  = val;
        val   = newval;
    }

    float getAcc(){
        return fabs(fabs(val-valp) - fabs(valp-valpp));
    }
};

class DimStat {
   //Dimension statistics
   public:
    using KeyT = int;

    DimStat() {}
    explicit DimStat(const KeyT& w) : dimId(w) {}
    virtual const KeyT& id() const { return dimId; }

    // Serialization and deserialization
    friend husky::BinStream& operator<<(husky::BinStream& stream, DimStat u) {
        stream << u.dimId << u.mean << u.var;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, DimStat& u) {
        stream >> u.dimId >> u.mean >> u.var;
        return stream;
    }

    int dimId;
    float mean;
    float var;
};


void pic() {
    husky::io::HDFSLineInputFormat infmt;
    infmt.set_input(husky::Context::get_param("input"));

    int dimension = std::stoi(husky::Context::get_param("dimension"));

    // 1. Create and globalize ndpoint objects
    husky::ObjList<NDpoint> ndpoint_list;
    auto parse_ndpoint = [&ndpoint_list](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator it = tok.begin();
        int id = stoi(*it++);
        std::vector<float> coords;
        while (it != tok.end()) {
            coords.push_back(stod(*it++));
        }
        ndpoint_list.add_object(NDpoint(id, coords));
    };
    load(infmt, parse_ndpoint);
    globalize(ndpoint_list);
    husky::base::log_msg( "ndpoint_list size " + std::to_string( ndpoint_list.get_size()));

    husky::base::log_msg( "part1 ok ");
    
    // 2. calculate the variance used in similarity function
    husky::ObjList<DimStat> dimstat_list;
    globalize(dimstat_list);

    husky::lib::Aggregator<int> sumAgg;
    auto& ac = husky::lib::AggregatorFactory::get_channel();

    auto& meanCh =
        husky::ChannelFactory::create_push_combined_channel<float, husky::SumCombiner<float>>(ndpoint_list, dimstat_list);
    auto& varCh =
        husky::ChannelFactory::create_push_combined_channel<float, husky::SumCombiner<float>>(ndpoint_list, dimstat_list);
    list_execute(ndpoint_list, {}, {&ac, &meanCh, &varCh}, [&sumAgg, &meanCh, &varCh](NDpoint& p) {
        int idx=0;
        for (auto& aVal : p.coords) {
            meanCh.push(      aVal, idx);
            varCh .push( aVal*aVal, idx);
            idx++;
        }
        sumAgg.update(1);
    });

    list_execute(dimstat_list, [&sumAgg, &meanCh, &varCh](DimStat& d) {
        int count = sumAgg.get_value(); 
        d.mean = meanCh.get(d)/count;
        d.var  = varCh.get(d)/count - d.mean*d.mean; //Var(x) = <x^2> - <x><x>
        husky::base::log_msg( "stat " + std::to_string(d.id()) + "  " + std::to_string(count) + " " + std::to_string(d.mean) + " " + std::to_string(d.var));
    });

    husky::base::log_msg( "part2 ok ");

    // 3. calculate the affinity matrix
    husky::ObjList<MatElem> affMatElem_list;
    globalize(affMatElem_list);
    auto& pairCh =
        husky::ChannelFactory::create_push_channel<NDpoint>(ndpoint_list, affMatElem_list);

    //FIXME:: seems stupid to create a Channel just to make a list accessable by all worker?
    auto& dimstatBCh = husky::ChannelFactory::create_broadcast_channel<int,DimStat>(dimstat_list);
    list_execute(dimstat_list, [&](DimStat& d) {
        dimstatBCh.broadcast( d.id(), d);
    });

    ////Why this doesn't work? I get vec of size 0 even dimstat_list is globalized
    //std::vector<DimStat> local_dimstat_list = dimstat_list.get_data(); 
    //husky::base::log_msg( "local dim stat size " + std::to_string(local_dimstat_list.size()));
    //assert(local_dimstat_list.size() == dimension);
    
    size_t totDataPoints = sumAgg.get_value();

    //FIXME:: how to decide point pairing up to minimize traffic? Will husky auto place affMatElem on worker with best data locality?
    //        right now I push 2 data points to each matrix element
    list_execute(ndpoint_list, [&](NDpoint & p) {
        for (uint i = 0; i < p.id(); ++i) {
            pairCh.push(p, i*totDataPoints + p.id() );
        }
        for (uint i = p.id()+1; i < totDataPoints; ++i) {
            pairCh.push(p, p.id()*totDataPoints + i );
        }
    });

    husky::base::log_msg( "affMat size " + std::to_string(affMatElem_list.get_size()));

    husky::ObjList<DimStat> rowstat_list;
    auto& rowsumCh =
        husky::ChannelFactory::create_push_combined_channel<float, husky::SumCombiner<float>>(affMatElem_list, rowstat_list);

    list_execute(affMatElem_list, [&](MatElem & e){
        auto pts = pairCh.get(e);
        float nSigmaSq = 0;

        //if(e.id()==10011){
        //    husky::base::log_msg( "========" );
        //    for (auto aVal : pts){ husky::base::log_msg( "getval " + std::to_string(aVal.coords[0]));}
        //    husky::base::log_msg( "========" + std::to_string(local_dimstat_list[0]->var) );
        //}

        for (uint i =0; i<dimension;i++){ nSigmaSq += pow( pts[0].coords[i] - pts[1].coords[i], 2)/ (2.*dimstatBCh.get(i).var); }
        e.val = exp( -nSigmaSq);
        int row = e.id()/totDataPoints;
        int col = e.id()%totDataPoints;
        rowsumCh.push( e.val, row);
        rowsumCh.push( e.val, col);
        rowsumCh.push( e.val, -1 ); //hack, -1 is for calculating the volume (sum of all elements) of the affMat
    });

    //calculate then boardcast the rowsum. (is there a push_combine_boardcast channel API?)
    auto& rowsumBCh = husky::ChannelFactory::create_broadcast_channel<int, float>(rowstat_list);
    list_execute(rowstat_list, [&](DimStat & r){
        rowsumBCh.broadcast( r.id(), rowsumCh.get(r) );
    });

    husky::base::log_msg( "part3 ok ");

    // 4. repeat multiply the affinity matrix to the vector till convergence

    //To row normalized the affMat we need to calculate D*A, where D is diagonal matrix with 1/rowsum at the diagonal
    //We want Vn = (D*A) * ... * (D*A) * (D*A) * V0
    //Instead of calculating D*A, we can just multiply D on the A*V0 vector
    //ie. V1 = D*(A*V0)
    //    V2 = D*(A*V1)...

    float affMatVol = rowsumBCh.get(-1)*2.;

    husky::ObjList<VElem> velem_list;
    int chunkSize = totDataPoints/husky::Context::get_worker_info()->get_num_workers()+1;
    int startPos  = chunkSize * husky::Context::get_global_tid();
    int endPos    = std::min( int(totDataPoints) , chunkSize * (husky::Context::get_global_tid()+1));

    //Init V0
    for (uint i = startPos; i<endPos; i++){
        VElem e(i);
        e.setVal( rowsumBCh.get(i) / affMatVol );
        velem_list.add_object( e );
    }
    globalize(velem_list);

    husky::base::log_msg( "vec "
                          + std::to_string(velem_list.get_size()) + " "
                          + std::to_string(startPos) + " "
                          + std::to_string(endPos) + " "
                        );

    //FIXME:: Instead of the whole object, should push only the val and id of VElem to the Mat
    auto& vecToMatCh =
        husky::ChannelFactory::create_push_channel<VElem>(velem_list, affMatElem_list);
    auto& matToVecCh =
        husky::ChannelFactory::create_push_combined_channel<float, husky::SumCombiner<float>>(affMatElem_list, velem_list);

    husky::lib::Aggregator<float> maxFloatAgg(0., [](float& a, const float& b){ if (b>a) a=b; } );

    int   maxIter = std::stoi(husky::Context::get_param("maxIter"));
    float epsilon = std::stod(husky::Context::get_param("stopThres"));

    if (maxIter<0) maxIter=10;
    if (epsilon<0) epsilon=1e-5/totDataPoints;

    for (uint iter=0; iter<maxIter; ++iter){

        list_execute(velem_list, [&](VElem & v){
            for (uint i = 0       ; i < v.id()       ; ++i) { vecToMatCh.push( v, i*totDataPoints + v.id() ); }
            for (uint i = v.id()+1; i < totDataPoints; ++i) { vecToMatCh.push( v, v.id()*totDataPoints + i ); }
        });

        list_execute(affMatElem_list, [&](MatElem & e){
            int row = e.id()/totDataPoints;
            int col = e.id()%totDataPoints;
            auto& vlist = vecToMatCh.get(e);
            if (vlist.size()!=2) husky::base::log_msg( "What? " + std::to_string(vlist.size()) );
            for (auto& aVElem : vlist){
                if      (aVElem.id()==row) matToVecCh.push( e.val*aVElem.val, col );
                else if (aVElem.id()==col) matToVecCh.push( e.val*aVElem.val, row );
                else husky::base::log_msg( "What2? " + std::to_string(aVElem.id()) + " " + std::to_string(e.id() ) );
            }
        });
        
        list_execute(velem_list,{&matToVecCh},{&ac}, [&](VElem & v){
            v.setVal( matToVecCh.get(v) / rowsumBCh.get( v.id() ) );
            maxFloatAgg.update( v.getAcc() );
        });
        
        if (husky::Context::get_global_tid()==0){
            husky::base::log_msg(   "Iter "   + std::to_string(iter) + " "
                                  + "MaxAcc " + std::to_string(maxFloatAgg.get_value()*totDataPoints ));
        }

        if (maxFloatAgg.get_value()<epsilon) break;
        maxFloatAgg.to_reset_each_iter();
        //maxFloatAgg.to_keep_aggregate();


    }//end iter loop

    //lazily output to screen
    list_execute(velem_list, [&](VElem & v){
        char s[100];
        sprintf(s, "%e", v.val);
        husky::base::log_msg( "result v "
                              + std::to_string(v.id() )   + " "
                              + s  + " "
                            );
    });


}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("dimension");
    args.push_back("maxIter");
    args.push_back("stopThres");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(pic);
        return 0;
    }
    return 1;
}
