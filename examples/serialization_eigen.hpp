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

#include "core/engine.hpp"
#include "Eigen/SparseCore"

namespace husky {
namespace base {

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVecT;

BinStream & operator >> (BinStream & in, SpMat & mat);
BinStream & operator << (BinStream & out, const SpMat & mat);
BinStream & operator >> (BinStream & in, Eigen::MatrixXd & mat);
BinStream & operator << (BinStream & out, const Eigen::MatrixXd & mat);
BinStream & operator >> (BinStream & in, Eigen::VectorXd & vec);
BinStream & operator << (BinStream & out, const Eigen::VectorXd & vec);
BinStream & operator >> (BinStream & in, SpVecT& vec);
BinStream & operator << (BinStream & out, const SpVecT & vec);

}  // namespace base
}  // namespace husky
