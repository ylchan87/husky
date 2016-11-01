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
#include <ctime>

#include "boost/tokenizer.hpp"

#include "core/engine.hpp"
#include "io/input/line_inputformat.hpp"
#include "lib/aggregator_factory.hpp"

using namespace std;

class NDpoint {
   //N-dimension point modelling the input data
   public:
    using KeyT = int;

    NDpoint() {}
    explicit NDpoint(const KeyT& w) : pointId(w) {}
    NDpoint(const KeyT& pId, std::vector<double> coords) {
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
    std::vector<double> coords;
};

void vrpca() {
    std::clock_t startT;
    double timeLapsed;

    husky::io::LineInputFormat infmt;
    infmt.set_input(husky::Context::get_param("input"));

    const int dimension  = std::stoi(husky::Context::get_param("dimension"));
    int nepoch           = std::stoi(husky::Context::get_param("nepoch"));
    int niter            = std::stoi(husky::Context::get_param("niter"));
    int nsyncPerEpoch    = std::stoi(husky::Context::get_param("nsyncPerEpoch"));
    double eta           = std::stod(husky::Context::get_param("eta"));

    husky::lib::Aggregator<vector<double>> uAgg(vector<double>(dimension),
                                                [](vector<double>& a, const vector<double>& b) {
                                                  for (int i = 0; i < a.size(); ++i) a[i] += b[i];
                                                },
                                                [&](vector<double>& v) {
                                                  v = std::move(vector<double>(dimension));
                                                }
                                               );

    husky::lib::Aggregator<vector<double>> wAgg(vector<double>(dimension),
                                                [](vector<double>& a, const vector<double>& b) {
                                                  //husky::base::log_msg( "agg" + std::to_string(a.size()) + " " + std::to_string(b.size()));
                                                  for (int i = 0; i < a.size(); ++i) a[i] += b[i];
                                                },
                                                [&](vector<double>& v) {
                                                  v = std::move(vector<double>(dimension));
                                                }
                                               );

    husky::lib::Aggregator<vector<double>> dataMeanAgg(vector<double>(dimension),
                                                       [](vector<double>& a, const vector<double>& b) {
                                                         for (int i = 0; i < a.size(); ++i) a[i] += b[i];
                                                       },
                                                       [&](vector<double>& v) {
                                                         v = std::move(vector<double>(dimension));
                                                       }
                                                      );

    husky::lib::Aggregator<double> totDataSqAgg;
    husky::lib::Aggregator<int> totRowAgg;

    husky::lib::Aggregator<double> residueAgg;//FIXME: validation only, to be removed

    auto& ac = husky::lib::AggregatorFactory::get_channel();


    // 1. Read in the data rows
    husky::ObjList<NDpoint> ndpoint_list;
    int localId = 0;
    auto parse_ndpoint = [&](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator it = tok.begin();
        int id = stoi(*it++);
        std::vector<double> coords;
        for (int i=0;i<dimension;++i){
            double val = stod(*it++);
            coords.push_back(val);
            totDataSqAgg.update(val*val);
        }
        dataMeanAgg.update(coords);
        totRowAgg.update(1);
        ndpoint_list.add_object(NDpoint(localId, coords));
        localId++;
    };

    startT = std::clock();
    load(infmt, {&ac}, parse_ndpoint);
    timeLapsed = ( std::clock() - startT ) / (double) CLOCKS_PER_SEC;

    husky::base::log_msg( "ndpoint_list size " + std::to_string( ndpoint_list.get_size()));
    husky::base::log_msg( "Read data ok, sec used: " + to_string(timeLapsed));
    
    //TODO: provide option to subtract data mean 
    //std::vector<double> dataMean(dataMeanAgg.get_value()); //need a copy to do division;
    //for (auto& e : dataMean){ e/=totRowAgg.get_value();}

    std::vector<double> w(dimension);
    std::vector<double> wTilde(dimension); wTilde[0] = 1.0;
    std::vector<double> luTilde(dimension); //local uTilde
    std::vector<double> uTilde(dimension); //global uTilde

    std::default_random_engine generator; generator.seed(1234); //FIXME: fix seed or not?
    std::uniform_int_distribution<int> distribution(0,ndpoint_list.get_size()-1);
    auto dice = std::bind ( distribution, generator );

    if (niter<=0) niter = ndpoint_list.get_size(); //paper recommendation
    if (eta<=0.) eta = sqrt( totRowAgg.get_value() ) / totDataSqAgg.get_value(); //paper recommendation
    if (nsyncPerEpoch<=0) nsyncPerEpoch = 1;
    
    int nsyncInterval = niter/nsyncPerEpoch+1;

    startT = std::clock();
    for (int iepoch=0;iepoch<nepoch;iepoch++){
      std::fill(luTilde.begin(), luTilde.end(), 0);
      uAgg.to_reset_each_iter();
      residueAgg.to_reset_each_iter(); //FIXME: validation only, to be removed

      //calculate uTilde
      list_execute(ndpoint_list, {}, {&ac}, [&](NDpoint& p) {
          double dot=0.;
          for (int i=0;i<dimension;i++) { dot += p.coords[i]*wTilde[i];}
          for (int i=0;i<dimension;i++) { luTilde[i] += dot* p.coords[i];}

          residueAgg.update(dot*dot);//FIXME: validation only, to be removed
      });


      std::ostringstream oss; oss.precision(17);
      oss << "epoch " << iepoch << " " << std::scientific << residueAgg.get_value();
      husky::base::log_msg( oss.str());//FIXME: validation only, to be removed

      uAgg.update(luTilde);
      husky::lib::AggregatorFactory::sync();
      uTilde = uAgg.get_value();
      for (auto& e : uTilde){ e/=totRowAgg.get_value();}

      //residueAgg.get_value() become 0.0 here ?!
      //husky::base::log_msg( "epoch " + to_string(iepoch) + " " + to_string(residueAgg.get_value()));//FIXME: validation only, to be removed

      //husky::base::log_msg( "uTilde ok");

      w = wTilde;
      for (int iiter=0;iiter<niter;++iiter){
        double wNorm = 0.;
        double tmp = 0.;
        auto& aRow = ndpoint_list.get( dice() ).coords;
        
        for (int i=0;i<dimension;++i){ tmp += aRow[i]*(w[i]-wTilde[i]);}
        for (int i=0;i<dimension;++i){
          w[i] += eta * ( aRow[i]*tmp + uTilde[i]);
          wNorm += w[i]*w[i];
        }
        
        wNorm = sqrt(wNorm);
        for (int i=0;i<dimension;++i){ w[i] /= wNorm;}
        
        //periodically combine steps from all workers
        if ((iiter+1) % nsyncInterval==0 || (iiter+1)==niter){
          double thisWorkerWeight = double(ndpoint_list.get_size())/totRowAgg.get_value()*husky::Context::get_worker_info()->get_num_workers();
          for (int i=0;i<dimension;++i){ w[i] *= thisWorkerWeight;}
          wAgg.to_reset_each_iter();
          wAgg.update(w);
          husky::lib::AggregatorFactory::sync();
          w = wAgg.get_value();

          wNorm = 0.;
          for (int i=0;i<dimension;++i){ wNorm += w[i]*w[i];} wNorm = sqrt(wNorm);
          for (int i=0;i<dimension;++i){ w[i] /= wNorm;}

        }
      }
      wTilde = w;


      //if (husky::Context::get_global_tid()==0){
      //  std::ostringstream oss;
      //  oss << "epoch " << iepoch << " ";
      //  oss << std::scientific; oss.precision(17);
      //  for (int i=0;i<dimension;++i){ oss << w[i] << " ";}
      //  oss << std::endl;
      //  husky::base::log_msg( oss.str() );
      //}
    }
    timeLapsed = ( std::clock() - startT ) / (double) CLOCKS_PER_SEC;
    husky::base::log_msg( "All epoch complete, sec used: " + to_string(timeLapsed));


}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("dimension");
    args.push_back("nepoch");
    args.push_back("niter");
    args.push_back("nsyncPerEpoch");
    args.push_back("eta");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(vrpca);
        return 0;
    }
    return 1;
}
