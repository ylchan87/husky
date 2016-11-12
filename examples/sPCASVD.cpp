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
//

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"
#include "Eigen/LU"
#include "Eigen/SparseCore"
#include "Eigen/SVD"

#include "core/engine.hpp"
#include "io/input/line_inputformat.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/vector.hpp"
#include "serialization_eigen.hpp"
#include "io/hdfs_manager.hpp"

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::MatrixXd Mat;


class SparseRowObject{
public:
    using KeyT = int;

    KeyT name;
    std::vector<std::pair<int, double>> temp_row;
    SpMat row;

    SparseRowObject(){}
    explicit SparseRowObject(const KeyT& s) : name(s) {}    
    virtual const KeyT & id() const { return name; }

    friend husky::BinStream & operator >> (husky::BinStream & stream, SparseRowObject & r) {
        stream >> r.name >> r.temp_row >> r.row;
        return stream;
    }
    friend husky::BinStream & operator << (husky::BinStream & stream, SparseRowObject & r) {
        stream << r.name << r.temp_row << r.row;
        return stream;
    }

    void key_init(KeyT name) {
        this->name = name;
    }
    void row_init(SpMat x) {
        row = x;
    }
};

void sPCASVD() {
    husky::io::LineInputFormat infmt;
    infmt.set_input(husky::Context::get_param("input"));

    std::mt19937 generator( stoi(husky::Context::get_param("randseed")) );
    std::normal_distribution<double> normDist(0.0,1.0);
    auto normDice = std::bind ( normDist, generator );

    int d = stoi(husky::Context::get_param("nComponent"));

    // Create and globalize row objects
    husky::lib::Aggregator<int> totRowAgg;
    husky::lib::Aggregator<int> maxFeatureAgg(0,[](int& a, const int& b){ if (a < b) { a = b; } });
    husky::lib::Aggregator<double> ss1Agg;
    husky::lib::Aggregator<double> feaSumAgg;
    auto& ac = husky::lib::AggregatorFactory::get_channel();

    auto& row_list = husky::ObjListFactory::create_objlist<SparseRowObject>();
    auto parse_row = [&](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        auto it = tok.begin();
        int id = stoi(*it++);
        SparseRowObject r(id);
        while (it != tok.end()) {
            boost::char_separator<char> sep2(":");
            boost::tokenizer<boost::char_separator<char>> tok2(*it, sep2);
            auto it2 = tok2.begin();
            int fea_index = std::stoi(*it2++);
            maxFeatureAgg.update(fea_index);
            double fea_val = std::stod(*it2++);
            ss1Agg.update(fea_val*fea_val);
            feaSumAgg.update( fabs(fea_val) );
            r.temp_row.push_back(std::make_pair(fea_index, fea_val));
            it++;
        }
        row_list.add_object(std::move(r));
        totRowAgg.update(1);
    };
    husky::load(infmt, {&ac}, parse_row);

    husky::base::log_msg("ss1 " + std::to_string(ss1Agg.get_value()));
    husky::base::log_msg("feaSum " + std::to_string(feaSumAgg.get_value()));

    long N = totRowAgg.get_value();
    int D = maxFeatureAgg.get_value()+1;
    husky::base::log_msg("No. of row " + std::to_string(N) );
    husky::base::log_msg("Dim of features " + std::to_string(D) );
    list_execute(row_list, [&](SparseRowObject & r) {
        SpMat sparse_row(1, D);
        sparse_row.reserve( r.temp_row.size());
        for (auto&  it : r.temp_row) {
            sparse_row.insert(0, it.first) = it.second;
        }
        r.row = std::move(sparse_row);
        r.temp_row.clear();
        r.temp_row.shrink_to_fit();
    });

    husky::globalize(row_list);

    // initialize ss
    double ss = normDice();

    // initialize C
    Mat C(D, d);
    for (int i=0; i<D; ++i){
    for (int j=0; j<d; ++j){
        C(i,j) = normDice();
    }}

    //ss1
    double ss1 = ss1Agg.get_value();

    Mat I_d = Eigen::MatrixXd::Identity(d, d);
    Mat zero_dd = Eigen::MatrixXd::Zero(d, d);
    Mat zero_Dd = Eigen::MatrixXd::Zero(D, d);
    husky::lib::Aggregator<Mat> XtX_agg(zero_dd, [](Mat & a, const Mat & b) { a += b; }, [&](Mat & a) { a = zero_dd; });
    husky::lib::Aggregator<Mat> YtX_agg(zero_Dd, [](Mat & a, const Mat & b) { a += b; }, [&](Mat & a) { a = zero_Dd; });
    std::map<int, Mat> Xrows;
    husky::lib::Aggregator<double> ss3Agg;
    husky::lib::Aggregator<double> reconErrAgg;

    // iterations
    double errLimit = stod(husky::Context::get_param("errLimit"));
    int maxIter = stod(husky::Context::get_param("maxIter"));
    int iter = 0;
    double reconErr = 1000.0;
    for (iter=0;iter<maxIter;++iter){
        // 1.  M = C' * C + ss * I
        Eigen::MatrixXd M(d, d);
        M = C.transpose() * C;
        M += ss * I_d;

        // 2. CM = C * M(-1)
        Eigen::MatrixXd CM(D, d);
        CM = C * M.inverse();

        // 3. XM = Ym * CM   no mean centering for SVD
        // Eigen::MatrixXd Xm(1, d);
        // Xm = Ym * CM;

        // 4. (parallel)
        //     X = Y * CM - Xm
        //     XtX = X' * X
        //     YtX = Y' * X - Ym' * X
        list_execute(row_list, {}, {&ac}, [&](SparseRowObject & r) {
            Mat Xrow(1, d);
            Xrow = r.row * CM;
            Xrows[r.id()] = Xrow;
            XtX_agg.update( Xrow.transpose() * Xrow );
            YtX_agg.update( r.row.transpose() * Xrow );
        });
        Mat XtX = XtX_agg.get_value();
        Mat YtX = YtX_agg.get_value();

        // 5.  XtX += ss * M(-1)
        XtX += ss * M.inverse();

        // 6.  C = YtX / Xtx
        C = YtX * XtX.inverse();
        Mat Ct = C.transpose();

        // 7.  ss2 = tr(XtX * C' * C)
        double ss2 = (XtX * Ct * C).trace();

        // 8. ss3 = sigma(n = 1 to N) Xn * C' * (Ycn)'
        list_execute(row_list, {}, {&ac}, [&](SparseRowObject & r) {
            Mat reconY(1, D);
            Mat reconYerr(1, D);
            reconY = Xrows[r.id()] * Ct;
            reconYerr = reconY - Mat(r.row);
            ss3Agg.update( (reconY * r.row.transpose()).sum() );
            // reconstructin error = (Yi - Ym) * CM * C' - (Yi - Ym)
            if (reconYerr.cwiseAbs().sum()>100){
              for (int i=0;i<D;i++){
                husky::base::log_msg("diff " + std::to_string(reconY(1,i)) + " " + std::to_string(r.row.coeffRef(1,i)));
              }
            }
            reconErrAgg.update( reconYerr.cwiseAbs().sum() );
        });
        double ss3 = ss3Agg.get_value();

        // 9. ss = (ss1 + ss2 - 2 * ss3) / N / D
        ss = (ss1 + ss2 - 2 * ss3) / N / D;
        reconErr = reconErrAgg.get_value() / feaSumAgg.get_value();
        if (husky::Context::get_global_tid() == 0) {
            husky::base::log_msg("This is the " + std::to_string(iter) + " th iteration");
            husky::base::log_msg("Reconstruction error is " + std::to_string(reconErr));
        }

        // ---------------------
        // Add myself, orthonormalize C
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
        C = svd.matrixU();
        // ---------------------

        XtX_agg.to_reset_each_iter();
        YtX_agg.to_reset_each_iter();
        ss3Agg.to_reset_each_iter();
        reconErrAgg.to_reset_each_iter();

        if( reconErr < errLimit) break;
    }

    // V is given by C, now Calculate U and S

    // extract S from unnormalized U, which is our PCA score Xrows
    typedef husky::lib::DenseVector<double> Vec;
    husky::lib::Aggregator<Vec> sValsAgg( Vec(d,0.), [](Vec & a, const Vec & b) { a += b; }, [&](Vec & a) { a = Vec(d,0.); });
    husky::lib::DenseVector<double> sVals(d,0);
    list_execute(row_list, {}, {&ac}, [&](SparseRowObject & r) {
        Mat Xrow(1, d);
        Xrow = r.row * C;
        Xrows[r.id()] = Xrow;
        for (int i=0;i<d;i++){ sVals[i] = Xrow(0,i)*Xrow(0,i); }
        sValsAgg.update( sVals );
    });

    sVals = sValsAgg.get_value();
    for (int i=0;i<d;i++){ 
      sVals[i] = sqrt(sVals[i]);
    }

    // normalize and output U
    for (auto& r : Xrows){
      for (int i=0;i<d; ++i){ r.second.coeffRef(i) /= sVals[i]; }
      std::ostringstream oss;
      oss.precision(10);
      oss << r.first << " ";
      for (int i=0;i<d; ++i){ oss << r.second.coeffRef(i) << " "; }
      oss << "\n";
      husky::io::HDFS::Write("master", "9000", oss.str(), husky::Context::get_param("outDir") + "/svd_u",
                             husky::Context::get_global_tid());
    }
    
    // output S and V
    if (husky::Context::get_global_tid() == 0) {
      husky::base::log_msg("After " + std::to_string(iter) + " iterations");
      husky::base::log_msg("ss is " + std::to_string(ss));
      husky::base::log_msg("recon error is " + std::to_string(reconErr));

      husky::base::log_msg("idx singular values v: ");
      
      std::ostringstream oss;
      for (int i=0;i<d;i++){ 
        oss << i << " " << sVals[i] << " ";
        for (int j=0;j<D;j++){ oss << C(j,i) << " ";}
        oss << "\n" ;
      }
      husky::base::log_msg(oss.str());
      husky::io::HDFS::Write("master", "9000", oss.str(), husky::Context::get_param("outDir") + "/svd_sv",
                             husky::Context::get_global_tid());
    }

}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("randseed");
    args.push_back("nComponent"); //no. of top singular values to get
    args.push_back("errLimit");
    args.push_back("maxIter");
    args.push_back("outDir");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(sPCASVD);
        return 0;
    }
    return 1;
}
