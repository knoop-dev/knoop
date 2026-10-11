/**
 * This file is part of knoop, copyright (C) 2025-2026 Kári Hlynsson.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <Eigen/Dense>
#include <boost/algorithm/string/case_conv.hpp>
#include <functional>

namespace knoop {

/**
 * Aggregators are functions that are applied to the phenotype values of a
 * subset of a proband's constituents to produce a single aggregate trait vector
 * that enters a weighting function.
 */
enum class AggFunction { Max, Min, Mean, Identity };

inline AggFunction aggregatorFromString(const std::string& s) {
  std::string l = boost::to_lower_copy(s);
  if (l == "max") return AggFunction::Max;
  if (l == "min") return AggFunction::Min;
  if (l == "mean") return AggFunction::Mean;
  if (l == "identity") return AggFunction::Identity;
  throw std::runtime_error("Unknown aggregator type" + s);
}

inline Eigen::MatrixXd aggregate(
    const Eigen::MatrixXd& preaggregate, AggFunction agg) {
  if (agg == AggFunction::Mean) return preaggregate.colwise().mean();
  if (agg == AggFunction::Max) return preaggregate.colwise().maxCoeff();
  if (agg == AggFunction::Min) return preaggregate.colwise().minCoeff();
  if (agg == AggFunction::Identity) {
    if (preaggregate.rows() > 1)
      throw std::runtime_error(
          std::format(
              "Identity aggregation can only be applied to single individual "
              "(got {})",
              preaggregate.rows()));
    return preaggregate.row(0);
  }
  throw std::invalid_argument("Unrecognised aggregation function");
}

/**
 * Weight functions are used to assign the probability of selection into a
 * sample based on some rule.
 */
using WeightFunction =
    std::function<void(const Eigen::MatrixXd&, Eigen::VectorXd&)>;

inline WeightFunction uniform() {
  return [](const Eigen::MatrixXd& /*agg*/, Eigen::VectorXd& res) {
    res.setConstant(1.0);
  };
}

inline WeightFunction logistic(const Eigen::VectorXd& effects) {
  return [effects](const Eigen::MatrixXd& agg, Eigen::VectorXd& res) {
    res = 1.0 / (1.0 + (-agg * effects).array().exp());
  };
}

inline WeightFunction caseControl(
    double threshold, bool above, const Eigen::VectorXd& effects) {
  return [threshold, above, effects](
             const Eigen::MatrixXd& agg, Eigen::VectorXd& res) {
    Eigen::VectorXd score = agg * effects;
    if (above) {
      res = (score.array() > threshold).cast<double>().matrix();
    } else {
      res = (score.array() < threshold).cast<double>().matrix();
    }
  };
}

};  // namespace knoop
