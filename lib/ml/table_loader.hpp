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

#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "boost/tokenizer.hpp"

#include "base/exception.hpp"
#include "core/attrlist.hpp"
#include "core/executor.hpp"
#include "core/objlist.hpp"
#include "core/utils.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/vector.hpp"

namespace husky {
namespace lib {
namespace ml {
namespace tableloader {

class Row {
   public:
    using KeyT = int;
    KeyT key;
    const KeyT& id() const { return key; }
    Row() = default;
    explicit Row(const KeyT& k) : key(k) {}
};

template <typename... colTypes>
class TableLoader {
   public:
    TableLoader();
    ~TableLoader();

    ObjList<Row>& load(std::string url);

    void set_col_names(std::vector<std::string> col_names);
    void set_col_defaults(std::tuple<colTypes...> col_defaults) { this->col_defaults = col_defaults; }
    void set_col_separators(std::string col_separators) { this->col_separators = col_separators; }

    void set_col_dim(int col_index, int dim);
    void set_col_dim(std::string col_name, int dim);

    void set_allow_parse_failure(bool b) { this->allow_parse_failure = b; }

    static const int col_num = sizeof...(colTypes);
    static const std::make_index_sequence<col_num> col_indexes;

   private:
    std::vector<husky::AttrListBase*> cols;
    std::tuple<colTypes...> col_defaults;
    std::vector<std::string> col_names;
    std::vector<int> col_dims;
    std::vector<husky::lib::Aggregator<int>*> col_dim_aggs;
    std::string col_separators;
    bool allow_parse_failure;

    // variables used in the template loop inside parse_cols
    boost::tokenizer<boost::char_separator<char>>::iterator t_it;
    boost::tokenizer<boost::char_separator<char>>::iterator t_end;
    size_t t_objIdx;

    template <typename T>
    bool preparse(int col_index, T*) {
        return true;
    }                                                           // do nothing
    bool preparse(int col_index, husky::lib::SparseVectorXd*);  // specialize for sparse vector to probe its dim

    template <std::size_t... Is>
    void parse_cols(std::index_sequence<Is...>);

    template <typename T>
    bool parse_col(int col_index, T& defaultVal);
    bool parse_col(int col_index, husky::lib::VectorXd& defaultVal);
    bool parse_col(int col_index, husky::lib::SparseVectorXd& defaultVal);
    bool parse_col(int col_index, std::string& defaultVal);

    template <typename T>
    bool postparse(int col_index, T*) {
        return true;
    }                                                            // do nothing
    bool postparse(int col_index, husky::lib::SparseVectorXd*);  // specicalize for sparse vector
};

template <typename... colTypes>
TableLoader<colTypes...>::TableLoader() {
    for (int i = 0; i < col_num; ++i) {
        col_names.push_back("column" + std::to_string(i));
        col_dims.push_back(0);
        col_dim_aggs.push_back(NULL);
    };
    col_separators = " \t";
    allow_parse_failure = false;
}

template <typename... colTypes>
TableLoader<colTypes...>::~TableLoader() {
    for (int i = 0; i < col_num; ++i) {
        if (col_dim_aggs[i])
            delete col_dim_aggs[i];
    };
}

template <typename... colTypes>
TableLoader<colTypes...> get_instance() {
    return TableLoader<colTypes...>();
}

TableLoader<int, husky::lib::SparseVectorXd> get_libsvm_instance() {
    return TableLoader<int, husky::lib::SparseVectorXd>();
}

template <typename... colTypes>
TableLoader<colTypes...> get_csv_instance() {
    TableLoader<colTypes...> loader;
    loader.set_col_separators(",");
    return loader;
}

template <typename... colTypes>
bool TableLoader<colTypes...>::preparse(int col_index, husky::lib::SparseVectorXd*) {
    if (col_dims[col_index] <= 0) {
        col_dim_aggs[col_index] = new husky::lib::Aggregator<int>(0, [](int& a, const int& b) { a = std::max(a, b); });
    }
    return true;
}

template <typename... colTypes>
bool TableLoader<colTypes...>::postparse(int col_index, husky::lib::SparseVectorXd*) {
    if (col_dims[col_index] <= 0) {
        col_dims[col_index] = col_dim_aggs[col_index]->get_value();
        auto& data = static_cast<husky::AttrList<Row, husky::lib::SparseVectorXd>*>(cols[col_index])->get_data();
        for (auto& aElem : data) {
            aElem.conservativeResize(col_dims[col_index]);
            aElem.data().squeeze();
        }
        husky::LOG_I << "Resize sparse column " << col_index << " to " << col_dim_aggs[col_index]->get_value()
                     << std::endl;
    }
    return true;
}

// a template for loop that invoke parse_col for each type in template arg colTypes
template <typename... colTypes>
template <std::size_t... Is>
void TableLoader<colTypes...>::parse_cols(std::index_sequence<Is...>) {
    std::vector<bool> success = {parse_col(Is, std::get<Is>(col_defaults))...};
}

// general parser for column
template <typename... colTypes>
template <typename T>
bool TableLoader<colTypes...>::parse_col(int col_index, T& defaultVal) {
    if (t_it == t_end)
        return false;
    T val;
    std::stringstream ss(*t_it);
    ss >> val;
    if (!ss) {
        if (allow_parse_failure) {
            val = defaultVal;
        } else {
            throw base::HuskyException("Parse failure for column " + col_names[col_index]);
        }
    }
    static_cast<husky::AttrList<Row, T>*>(cols[col_index])->set(t_objIdx, val);
    t_it++;
    return true;
}

// specialize parser for column of Eigen lib vector
template <typename... colTypes>
bool TableLoader<colTypes...>::parse_col(int col_index, husky::lib::VectorXd& defaultVec) {
    if (t_it == t_end)
        return false;

    std::stringstream ss;
    husky::lib::VectorXd vec;
    double tmpval;
    if (col_dims[col_index] <= 0) {
        // if no dimension given, consume all data and update dimension
        int col_dim = 0;
        std::vector<double> tmpvec;
        while (t_it != t_end) {
            ss.clear();
            ss.str(*t_it);
            ss >> tmpval;
            if (!ss)
                break;
            tmpvec.push_back(tmpval);
            col_dim++;
            t_it++;
        }
        vec.resize(col_dim);
        col_dims[col_index] = col_dim;
        for (int i = 0; i < col_dim; i++) {
            vec[i] = tmpvec[i];
        };
    } else {
        vec.resize(col_dims[col_index]);
        for (int i = 0; i < col_dims[col_index]; i++) {
            ss.clear();
            ss.str(*t_it++);
            ss >> tmpval;
            if (!ss) {
                if (allow_parse_failure && i < defaultVec.size()) {
                    tmpval = defaultVec[i];
                } else {
                    throw base::HuskyException("Parse failure for column " + col_names[col_index]);
                }
            }
            vec[i] = tmpval;
        };
    }
    static_cast<husky::AttrList<Row, husky::lib::VectorXd>*>(cols[col_index])->set(t_objIdx, std::move(vec));
}

// specialize parser for column of Eigen lib sparse vector
template <typename... colTypes>
bool TableLoader<colTypes...>::parse_col(int col_index, husky::lib::SparseVectorXd& defaultVec) {
    if (t_it == t_end)
        return false;

    std::stringstream ss;
    husky::lib::SparseVectorXd vec(col_dims[col_index] > 0 ? col_dims[col_index] : INT_MAX);
    while (t_it != t_end) {
        int splitPos = t_it->find(":");
        if (splitPos == std::string::npos)
            break;

        bool success = true;

        ss.clear();
        ss.str(t_it->substr(0, splitPos));
        int idx;
        ss >> idx;
        if (!ss)
            success = false;

        ss.clear();
        ss.str(t_it->substr(splitPos + 1, t_it->length() - splitPos));
        double tmpval;
        ss >> tmpval;
        if (!ss)
            success = false;

        if (!success) {
            if (allow_parse_failure) {
                husky::LOG_I << "Skipped problematic token in parsing sparse vector column " << col_names[col_index];
                t_it++;
                continue;
            } else {
                throw base::HuskyException("Parse failure for column " + col_names[col_index]);
            }
        }

        if (idx + 1 >= col_dims[col_index]) {
            if (col_dims[col_index] > 0) {
                // user has provided dimension
                throw husky::base::HuskyException("vector dimension larger then user specified for column " +
                                                  col_names[col_index]);
            } else {
                // user has not provided dimension, update the largest dimension seen so far
                col_dim_aggs[col_index]->update(idx + 1);
            }
        }
        vec.coeffRef(idx) = tmpval;
        t_it++;
    }
    static_cast<husky::AttrList<Row, husky::lib::SparseVectorXd>*>(cols[col_index])->set(t_objIdx, std::move(vec));
}

// specialize parser for column of string
template <typename... colTypes>
bool TableLoader<colTypes...>::parse_col(int col_index, std::string& defaultVal) {
    if (t_it == t_end)
        return false;
    static_cast<husky::AttrList<Row, std::string>*>(cols[col_index])->set(t_objIdx, *t_it);
    t_it++;
    return true;
}

template <typename... colTypes>
ObjList<Row>& TableLoader<colTypes...>::load(std::string url) {
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(url);

    auto& table = ObjListStore::create_objlist<Row>();

    int tmpIdx;
    std::vector<bool> success;

    tmpIdx = 0;
    cols = {&table.create_attrlist<colTypes>(col_names[tmpIdx++])...};

    tmpIdx = 0;
    success = {preparse(tmpIdx++, (colTypes*) 0)...};

    husky::lib::Aggregator<int> nrows_agg;
    auto& ac = AggregatorFactory::get_channel();

    std::function<void(boost::string_ref)> readline;
    readline = [&](boost::string_ref line) {
        boost::char_separator<char> sep(col_separators.c_str());
        boost::tokenizer<boost::char_separator<char>> tok(line, sep);
        t_it = tok.begin();
        t_end = tok.end();
        t_objIdx = table.add_object(Row());

        nrows_agg.update(1);
        parse_cols(col_indexes);
    };
    husky::load(infmt, {&ac}, readline);

    tmpIdx = 0;
    success = {postparse(tmpIdx++, (colTypes*) 0)...};

    husky::LOG_I << "Total rows " << nrows_agg.get_value() << std::endl;

    return table;
}

template <typename... colTypes>
void TableLoader<colTypes...>::set_col_names(std::vector<std::string> col_names) {
    if (col_names.size() != col_num)
        throw base::HuskyException("Mismatch in number of columns");
    this->col_names = col_names;
}

template <typename... colTypes>
void TableLoader<colTypes...>::set_col_dim(int col_index, int dim) {
    if (col_index >= col_num || col_index < 0)
        throw base::HuskyException("Col index out of range");
    this->col_dims[col_index] = dim;
}

template <typename... colTypes>
void TableLoader<colTypes...>::set_col_dim(std::string col_name, int dim) {
    for (int i = 0; i < col_num; ++i) {
        if (col_names[i].compare(col_name) == 0) {
            col_dims[i] = dim;
            return;
        }
    }
    throw base::HuskyException("Column with given name not found");
    return;
}

}  // namespace tableloader
}  // namespace ml
}  // namespace lib
}  // namespace husky
