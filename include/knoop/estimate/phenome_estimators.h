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
#include <optional>

namespace knoop {

namespace details {

class EstimatorHeritabilityStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorHeritabilityStrategy(const Params& params)
      : PopulationEstimatorStrategy(
            "pheno_h2", {}, params.pheno.names, params.pheno.n_pheno) {
    n_pheno_ = params.pheno.n_pheno;
  }

  void compute(const State& state) override {
    for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno)
      data_(pheno) = state.pheno().componentVar(pheno, Component::Genetic) /
                     state.pheno().componentVar(pheno, Component::Total);
  }

 private:
  std::size_t n_pheno_;
};

class EstimatorComponentMeanStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorComponentMeanStrategy(const Params& params, Component type)
      : PopulationEstimatorStrategy(
            "pheno_" + componentToString(type) + "_mean",
            {},
            params.pheno.names,
            params.pheno.n_pheno),
        type_(type),
        n_pheno_(params.pheno.n_pheno) {}

  void compute(const State& state) override {
    for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno)
      data_(pheno) = state.pheno().componentMean(pheno, type_);
  }

 private:
  Component type_;
  std::size_t n_pheno_;
};

class EstimatorComponentVarStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorComponentVarStrategy(const Params& params, Component type)
      : PopulationEstimatorStrategy(
            "pheno_" + componentToString(type) + "_var",
            {},
            params.pheno.names,
            params.pheno.n_pheno),
        type_(type),
        n_pheno_(params.pheno.n_pheno) {}

  void compute(const State& state) override {
    for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno)
      data_(pheno) = state.pheno().componentVar(pheno, type_);
  }

 private:
  Component type_;
  std::size_t n_pheno_;
};

class EstimatorComponentCorStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorComponentCorStrategy(
      const Params& params, Component type_l, std::optional<Component> type_r)
      : PopulationEstimatorStrategy(
            "pheno_" + componentToString(type_l) + "_" +
                componentToString(type_r.value_or(type_l)) + "_cor",
            utils::vectorSuffix(
                params.pheno.names, "_" + componentToString(type_l)),
            utils::vectorSuffix(
                params.pheno.names,
                "_" + componentToString(type_r.value_or(type_l))),
            params.pheno.n_pheno,
            params.pheno.n_pheno),
        n_ind_(params.global.n_ind),
        n_pheno_(params.pheno.n_pheno),
        type_l_(type_l),
        type_r_(type_r.value_or(type_l)),
        std_l_(n_ind_, n_pheno_),
        std_r_(n_ind_, n_pheno_) {}

  void compute(const State& state) override {
    utils::standardise(state.pheno()(type_l_), std_l_);
    utils::standardise(state.pheno()(type_r_), std_r_);
    data_ = (std_l_.transpose() * std_r_) / static_cast<double>(n_ind_);
  }

 private:
  std::size_t n_ind_;
  std::size_t n_pheno_;
  Component type_l_;
  Component type_r_;
  Eigen::MatrixXd std_l_;
  Eigen::MatrixXd std_r_;
};

class EstimatorComponentCovStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorComponentCovStrategy(
      const Params& params, Component type_l, std::optional<Component> type_r)
      : PopulationEstimatorStrategy(
            "pheno_" + componentToString(type_l) + "_" +
                componentToString(type_r.value_or(type_l)) + "_cov",
            utils::vectorSuffix(
                params.pheno.names, "_" + componentToString(type_l)),
            utils::vectorSuffix(
                params.pheno.names,
                "_" + componentToString(type_r.value_or(type_l))),
            params.pheno.n_pheno,
            params.pheno.n_pheno),
        n_ind_(params.global.n_ind),
        n_pheno_(params.pheno.n_pheno),
        type_l_(type_l),
        type_r_(type_r.value_or(type_l)),
        centred_l_(n_ind_, n_pheno_),
        centred_r_(n_ind_, n_pheno_) {}

  void compute(const State& state) override {
    utils::standardise(state.pheno()(type_l_), centred_l_, true, true, false);
    utils::standardise(state.pheno()(type_r_), centred_r_, true, true, false);
    data_ =
        (centred_l_.transpose() * centred_r_) / static_cast<double>(n_ind_ - 1);
  }

 private:
  std::size_t n_ind_;
  std::size_t n_pheno_;
  Component type_l_;
  Component type_r_;
  Eigen::MatrixXd centred_l_;
  Eigen::MatrixXd centred_r_;
};

}  // namespace details

inline PopulationEstimator populationHeritability() {
  return PopulationEstimator{
      .name = "heritability", .fn = [](const Params& params) {
        return std::make_unique<details::EstimatorHeritabilityStrategy>(params);
      }};
}

inline PopulationEstimator populationComponentMean(
    Component type = Component::Total) {
  return PopulationEstimator{
      .name = "pheno-mean-" + componentToString(type),
      .fn = [type](const Params& params) {
        return std::make_unique<details::EstimatorComponentMeanStrategy>(
            params, type);
      }};
}

inline PopulationEstimator populationComponentVar(
    Component type = Component::Total) {
  return PopulationEstimator{
      .name = "pheno-var-" + componentToString(type),
      .fn = [type](const Params& params) {
        return std::make_unique<details::EstimatorComponentVarStrategy>(
            params, type);
      }};
}

inline PopulationEstimator populationComponentCor(
    Component type_l = Component::Total,
    std::optional<Component> type_r = std::nullopt) {
  return PopulationEstimator{
      .name = "pheno-cor-" + componentToString(type_l) + "-" +
              componentToString(type_r.value_or(type_l)),
      .fn = [type_l, type_r](const Params& params) {
        return std::make_unique<details::EstimatorComponentCorStrategy>(
            params, type_l, type_r);
      }};
}

inline PopulationEstimator populationComponentCov(
    Component type_l = Component::Total,
    std::optional<Component> type_r = std::nullopt) {
  return PopulationEstimator{
      .name = "pheno-cov-" + componentToString(type_l) + "-" +
              componentToString(type_r.value_or(type_l)),
      .fn = [type_l, type_r](const Params& params) {
        return std::make_unique<details::EstimatorComponentCovStrategy>(
            params, type_l, type_r);
      }};
}

}  // namespace knoop
