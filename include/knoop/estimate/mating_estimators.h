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

#include <knoop/core/params.h>
#include <knoop/core/state.h>
#include <knoop/core/utils.h>
#include <knoop/estimate/estimator.h>
#include <knoop/estimate/strategy.h>

#include <Eigen/Dense>

namespace knoop {

namespace details {

class EstimatorMateCorStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorMateCorStrategy(const Params& params, Component type)
      : PopulationEstimatorStrategy(
            "mate_" + componentToString(type) + "_cor",
            utils::vectorSuffix(params.pheno.names, "_male"),
            utils::vectorSuffix(params.pheno.names, "_female"),
            params.pheno.n_pheno,
            params.pheno.n_pheno),
        type_(type),
        n_sex_(params.global.n_ind / 2),
        n_pheno_(params.pheno.n_pheno),
        std_male_(n_sex_, n_pheno_),
        std_female_(n_sex_, n_pheno_) {}

  void compute(const State& state) override {
    utils::standardise(state.pheno().male(type_), std_male_);
    utils::standardise(state.pheno().female(type_), std_female_);
    data_.setZero();

    for (std::size_t pair = 0; pair < n_sex_; ++pair)
      data_.noalias() += std_male_.row(pair).transpose() *
                         std_female_.row(state.matching()[pair]);
    data_ /= static_cast<double>(n_sex_);
  }

 private:
  Component type_;
  std::size_t n_sex_;
  std::size_t n_pheno_;
  Eigen::MatrixXd std_male_;
  Eigen::MatrixXd std_female_;
};

}  // namespace details

inline PopulationEstimator populationMateCor(
    Component type = Component::Total) {
  return PopulationEstimator{
      .name = "mate-cor-" + componentToString(type),
      .fn = [type](const Params& params) {
        return std::make_unique<details::EstimatorMateCorStrategy>(
            params, type);
      }};
}

}  // namespace knoop
