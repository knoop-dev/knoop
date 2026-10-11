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

#include <knoop/core/log.h>
#include <knoop/core/params.h>
#include <knoop/core/state.h>
#include <knoop/core/utils.h>
#include <knoop/data/pedigree.h>
#include <knoop/estimate/estimator.h>
#include <knoop/estimate/strategy.h>

#include <Eigen/Dense>
#include <deque>
#include <limits>

namespace knoop {

namespace details {

class EstimatorSiblingCovStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorSiblingCovStrategy(
      const Params& params, Component type, bool scale = false)
      : type_(type),
        scale_(scale),
        n_ind_(params.global.n_ind),
        n_pheno_(params.pheno.n_pheno),
        self_(params.global.n_ind, params.pheno.n_pheno),
        sibling_(params.global.n_ind, params.pheno.n_pheno),
        PopulationEstimatorStrategy(
            std::format(
                "sibling_{}_{}",
                componentToString(type),
                scale ? "cor" : "cov"),
            utils::vectorPrefix(params.pheno.names, "self_"),
            utils::vectorPrefix(params.pheno.names, "sibling_"),
            params.pheno.n_pheno,
            params.pheno.n_pheno) {}

  void compute(const State& state) override {
    std::vector<std::vector<Individual>> siblings =
        state.pedigree.getSiblings();

    auto buf = state.pheno()(type_);
    for (std::size_t ind = 0; ind < n_ind_; ++ind) {
      auto self = siblings[ind].front().index;
      auto sibling = siblings[ind].back().index;
      self_.row(ind) = buf.row(self);
      sibling_.row(ind) = buf.row(sibling);
    }

    utils::standardise(self_, self_, true, true, scale_);
    utils::standardise(sibling_, sibling_, true, true, scale_);

    data_ = (self_.transpose() * sibling_) / static_cast<double>(n_ind_ - 1);
  }

 private:
  Component type_;
  bool scale_;
  std::size_t n_ind_;
  std::size_t n_pheno_;
  Eigen::MatrixXd self_;
  Eigen::MatrixXd sibling_;
};

class EstimatorCousinCovStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorCousinCovStrategy(
      const Params& params,
      std::size_t degree,
      Component type,
      bool scale = false)
      : type_(type),
        scale_(scale),
        degree_(degree),
        n_ind_(params.global.n_ind),
        n_pheno_(params.pheno.n_pheno),
        self_(
            params.global.n_ind * (1ULL << (2 * degree)), params.pheno.n_pheno),
        cousin_(
            params.global.n_ind * (1ULL << (2 * degree)), params.pheno.n_pheno),
        PopulationEstimatorStrategy(
            std::format(
                "cousin_{}_{}_{}",
                degree,
                componentToString(type),
                scale ? "cor" : "cov"),
            utils::vectorPrefix(params.pheno.names, "self_"),
            utils::vectorPrefix(params.pheno.names, "cousin_"),
            params.pheno.n_pheno,
            params.pheno.n_pheno) {
    if (params.global.pedigree_max_depth < degree_ + 1) {
      throw std::invalid_argument(
          std::format(
              "Pedigree depth {} is insufficient to compute cousin "
              "covariance of degree {}; --pedigree-max-depth must be at least "
              "{}",
              params.global.pedigree_max_depth,
              degree_,
              degree_ + 1));
    }
  }

  void compute(const State& state) override {
    if (state.pedigree.size() < degree_ + 2) {
      data_.setConstant(std::numeric_limits<double>::quiet_NaN());
      return;
    }

    std::vector<std::vector<Individual>> paths =
        state.pedigree.getCousins(degree_);

    auto buf = state.pheno()(type_);
    for (std::size_t ind = 0; ind < paths.size(); ++ind) {
      auto self = paths[ind].front().index;
      auto cousin = paths[ind].back().index;
      self_.row(ind) = buf.row(self);
      cousin_.row(ind) = buf.row(cousin);
    }

    utils::standardise(self_, self_, true, true, scale_);
    utils::standardise(cousin_, cousin_, true, true, scale_);

    data_ =
        (self_.transpose() * cousin_) / static_cast<double>(paths.size() - 1);
  }

 private:
  Component type_;
  bool scale_;
  std::size_t degree_;
  std::size_t n_ind_;
  std::size_t n_pheno_;
  Eigen::MatrixXd self_;
  Eigen::MatrixXd cousin_;
};

// TODO: rework this once arbitrarily deep states enter simulation
class EstimatorAncestorCovStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorAncestorCovStrategy(
      const Params& params,
      std::size_t degree,
      Component type,
      bool scale = false)
      : type_(type),
        degree_(degree),
        scale_(scale),
        n_ind_(params.global.n_ind),
        n_pheno_(params.pheno.n_pheno),
        self_(params.global.n_ind * (1ULL << degree), params.pheno.n_pheno),
        ancestor_(params.global.n_ind * (1ULL << degree), params.pheno.n_pheno),
        PopulationEstimatorStrategy(
            std::format(
                "ancestor_{}_{}_{}",
                degree,
                componentToString(type),
                scale ? "cor" : "cov"),
            utils::vectorPrefix(params.pheno.names, "self_"),
            utils::vectorPrefix(params.pheno.names, "ancestor_"),
            params.pheno.n_pheno,
            params.pheno.n_pheno) {
    if (params.global.pedigree_max_depth < degree_) {
      throw std::invalid_argument(
          std::format(
              "Pedigree depth {} is insufficient to compute ancestor "
              "covariance of degree {}; --pedigree-max-depth must be at least "
              "{}",
              params.global.pedigree_max_depth,
              degree_,
              degree_));
    }
  }

  void compute(const State& state) override {
    syncPhenotypes(state);

    // needs history_.size() for now, not pedigree depth since this estimator
    // maintains its own phenotype buffer
    if (history_.size() < degree_ + 1 || state.pedigree.size() < degree_ + 1) {
      Log::debug(
          "Setting ancestral covariance estimator of degree {} to NaN in "
          "generation {}; pedigree "
          "size {}, max depth {}; history size {}",
          degree_,
          state.gen,
          state.pedigree.size(),
          state.pedigree.maxSize(),
          history_.size());

      data_.setConstant(std::numeric_limits<double>::quiet_NaN());
      return;
    }

    std::vector<std::vector<Individual>> paths =
        state.pedigree.getAncestors(degree_);

    auto self_buf = state.pheno()(type_);
    auto ancestor_buf = history_[degree_](type_);

    for (std::size_t ind = 0; ind < paths.size(); ++ind) {
      auto self = paths[ind].front().index;
      auto ancestor = paths[ind].back().index;
      self_.row(ind) = self_buf.row(self);
      ancestor_.row(ind) = ancestor_buf.row(ancestor);
    }

    utils::standardise(self_, self_, true, true, scale_);
    utils::standardise(ancestor_, ancestor_, true, true, scale_);

    data_ =
        (self_.transpose() * ancestor_) / static_cast<double>(paths.size() - 1);
  }

 private:
  Component type_;
  std::size_t degree_;
  bool scale_;
  std::size_t n_ind_;
  std::size_t n_pheno_;
  std::deque<PhenotypeBuffer> history_;
  Eigen::MatrixXd self_;
  Eigen::MatrixXd ancestor_;

  void syncPhenotypes(const State& state);
};

inline void EstimatorAncestorCovStrategy::syncPhenotypes(const State& state) {
  if (state.gen == 0) {
    history_.emplace_front(state.pheno(Generation::Parents));
    history_.emplace_front(state.pheno(Generation::Current));
  } else {
    if (history_.size() == degree_ + 1) history_.pop_back();
    history_.emplace_front(state.pheno(Generation::Current));
  }
}

}  // namespace details

inline PopulationEstimator populationSiblingCov(
    Component type = Component::Total) {
  return PopulationEstimator{
      .name = std::format("sibling-cov-", componentToString(type)),
      .fn = [type](const Params& params) {
        return std::make_unique<details::EstimatorSiblingCovStrategy>(
            params, type);
      }};
}

inline PopulationEstimator populationSiblingCor(
    Component type = Component::Total) {
  return PopulationEstimator{
      .name = std::format("sibling-cor-{}", componentToString(type)),
      .fn = [type](const Params& params) {
        return std::make_unique<details::EstimatorSiblingCovStrategy>(
            params, type, true);
      }};
}

inline PopulationEstimator populationCousinCov(
    std::size_t degree = 1, Component type = Component::Total) {
  return PopulationEstimator{
      .name = "cousin-" + std::to_string(degree) + "-cov-" +
              componentToString(type),
      .fn = [degree, type](const Params& params) {
        return std::make_unique<details::EstimatorCousinCovStrategy>(
            params, degree, type);
      }};
}

inline PopulationEstimator populationCousinCor(
    std::size_t degree = 1, Component type = Component::Total) {
  return PopulationEstimator{
      .name = "cousin-" + std::to_string(degree) + "-cor-" +
              componentToString(type),
      .fn = [degree, type](const Params& params) {
        return std::make_unique<details::EstimatorCousinCovStrategy>(
            params, degree, type, true);
      }};
}

inline PopulationEstimator populationAncestorCov(
    std::size_t degree = 1, Component type = Component::Total) {
  return PopulationEstimator{
      .name = "ancestor-" + std::to_string(degree) + "-cov-" +
              componentToString(type),
      .fn = [degree, type](const Params& params) {
        return std::make_unique<details::EstimatorAncestorCovStrategy>(
            params, degree, type);
      }};
}

inline PopulationEstimator populationAncestorCor(
    std::size_t degree = 1, Component type = Component::Total) {
  return PopulationEstimator{
      .name = "ancestor-" + std::to_string(degree) + "-cor-" +
              componentToString(type),
      .fn = [degree, type](const Params& params) {
        return std::make_unique<details::EstimatorAncestorCovStrategy>(
            params, degree, type, true);
      }};
}

}  // namespace knoop
