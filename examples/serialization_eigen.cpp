
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

#pragma once

#include "serialization_eigen.hpp"

namespace husky {
namespace base {

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVecT;

BinStream & operator >> (BinStream & in, SpMat & mat) {
    size_t innerSize, outerSize, rows, cols, nonZeros;
    in >> innerSize >> outerSize >> rows >> cols >> nonZeros;
    mat = SpMat(rows, cols);
    int row;
    int col;
    double val;
    while (nonZeros--) {
        in >> val >> row >> col;
        mat.coeffRef(row, col) = val;
    }
    return in;
}

BinStream & operator << (BinStream & out, const SpMat & mat) {
    out << size_t(mat.innerSize()) << size_t(mat.outerSize())
        << size_t(mat.rows()) << size_t(mat.cols())
        << size_t(mat.nonZeros());
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (SpMat::InnerIterator it(mat, k); it; ++it) {
            out << double(it.value()) << int(it.row()) << int(it.col());
        }
    }
    return out;
}

BinStream & operator >> (BinStream & in, Eigen::MatrixXd & mat) {
    size_t len_row, len_col;
    in >> len_row >> len_col;
    mat = Eigen::MatrixXd(len_row, len_col);
    double val;
    for (int i = 0; i < len_row; i++) {
        for (int j = 0; j < len_col; j++) {
            in >> i >> j >> val;
            mat.coeffRef(i, j) = val;
        }
    }
    return in;
}

BinStream & operator << (BinStream & out, const Eigen::MatrixXd & mat) {
    out << size_t(mat.rows());
    out << size_t(mat.cols());
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            out << i << j << double(mat(i, j));
        }
    }
    return out;
}

BinStream & operator >> (BinStream & in, Eigen::VectorXd & vec) {
    size_t len;
    in >> len;
    vec.resize(len);
    double val;
    for (size_t i = 0; i < len; i++) {
        in >> val;
        vec.coeffRef(i) = val;
    }
    return in;
}

BinStream & operator << (BinStream & out, const Eigen::VectorXd & vec) {
    size_t len = vec.size();
    out << len;
    for (size_t i = 0; i < len; i++) {
        out << double(vec(i));
    }
    return out;
}

BinStream & operator >> (BinStream & in, SpVecT& vec) {
    size_t innerSize, outerSize, len, nonZeros;
    in >> innerSize >> outerSize >> len >> nonZeros;
    vec = SpVecT(len);
    int index;
    double val;
    while (nonZeros--) {
        in >> val >> index;
        vec.coeffRef(index) = val;
    }
    return in;
}

BinStream & operator << (BinStream & out, const SpVecT & vec) {
    out << size_t(vec.innerSize()) << size_t(vec.outerSize())
        << size_t(vec.size())
        << size_t(vec.nonZeros());
    for (SpVecT::InnerIterator it(vec); it; ++it) {
        out << double(it.value()) << int(it.index());
    }
    return out;
}
// End serialization for Eigen

}  // namespace base
}  // namespace husky
