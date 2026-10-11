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
#include <knoop/core/random.h>
#include <knoop/core/state.h>

#include <ranges>
#include <stdexcept>
#include <vector>

namespace knoop {

/**
 * Score and compose phenotype components.
 */
class ScorePhenotypes {
 public:
  /**
   * @brief Construct a ScorePhenotypes object.
   *
   * @param params Parameter specification
   */
  explicit ScorePhenotypes(const Params& params)
      : n_ind_(params.global.n_ind),
        n_sex_(n_ind_ / 2),
        n_pheno_(params.pheno.n_pheno),
        tile_size_(
            std::ranges::max(
                params.pheno.pheno_loc |
                std::views::transform(&std::vector<std::size_t>::size))),
        pheno_effects_(params.pheno.pheno_effects),
        pheno_loc_(params.pheno.pheno_loc),
        var_env_(params.pheno.var_env),
        var_vert_(params.pheno.var_vert),
        rnur_pat_(params.pheno.rnur_pat),
        rnur_env_(params.pheno.rnur_env),
        vert_pat_(params.pheno.vert_pat),
        env_cor_(params.pheno.env_cor),
        env_chol_(env_cor_.llt().matrixL()),
        pheno_father_(n_sex_, n_pheno_),
        pheno_mother_(n_sex_, n_pheno_),
        nur_scale_(n_pheno_) {}

  void operator()(State& state);

 private:
  const std::size_t n_ind_;
  const std::size_t n_sex_;
  const std::size_t n_pheno_;
  const std::size_t tile_size_;

  const std::vector<Eigen::VectorXd>& pheno_effects_;
  const std::vector<std::vector<std::size_t>>& pheno_loc_;
  const Eigen::VectorXd& var_env_;
  const Eigen::VectorXd& var_vert_;
  const Eigen::VectorXd& rnur_pat_;
  const Eigen::VectorXd& rnur_env_;
  const Eigen::VectorXd& vert_pat_;
  const Eigen::MatrixXd& env_cor_;

  static constexpr std::size_t kIndTile = 512;
  Eigen::MatrixXd env_chol_;
  Eigen::MatrixXd pheno_father_;
  Eigen::MatrixXd pheno_mother_;
  Eigen::VectorXd nur_scale_;

  void scoreGenetic(State& state);
  void scoreEnvironmental(State& state);
  void scoreVertical(State& state);
  static void scoreTotal(State& state);
};

inline void ScorePhenotypes::scoreGenetic(State& state) {
  if (state.geno().view() != BufferLayout::LocusMajor)
    throw std::runtime_error("phenotype scoring requires loc-major view");

  auto& geno_buf = state.geno();
  auto pheno_gen_buf = state.pheno()(Component::Genetic);

#pragma omp parallel
  {
    Eigen::MatrixXd gen_tile(kIndTile, tile_size_);

#pragma omp for collapse(2) schedule(static)
    for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno) {
      for (std::size_t ind_tile = 0; ind_tile < n_ind_; ind_tile += kIndTile) {
        std::size_t tile_size = std::min(kIndTile, n_ind_ - ind_tile);

        // decompress the standardised genotype matrix in manageable tiles
        geno_buf.decompress(
            ind_tile,
            ind_tile + tile_size,
            pheno_loc_[pheno],
            gen_tile,
            true,
            true);

        // update the genetic component buffer in-place
        pheno_gen_buf.col(pheno).segment(ind_tile, tile_size).noalias() =
            gen_tile.topRows(tile_size).leftCols(pheno_loc_[pheno].size()) *
            pheno_effects_[pheno];
      }
    }
  }
}

inline void ScorePhenotypes::scoreEnvironmental(State& state) {
  auto pheno_env_buf = state.pheno()(Component::Environmental);

  knoop::random::Normal::sample(
      random::Xoshiro256pp::getInstance(),
      pheno_env_buf.data(),
      n_ind_ * n_pheno_);

  pheno_env_buf =
      pheno_env_buf * env_chol_.transpose() * var_env_.cwiseSqrt().asDiagonal();
}

inline void ScorePhenotypes::scoreVertical(State& state) {
  if (var_vert_.isZero(0)) return;

  utils::standardise(
      state.pheno(Generation::Parents).male(Component::Total), pheno_father_);
  utils::standardise(
      state.pheno(Generation::Parents).female(Component::Total), pheno_mother_);

  auto phenos_off = state.pheno(Generation::Current)(Component::Vertical);
  auto& match = state.matching(Generation::Parents);

  // score siblings
  for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno) {
    double vert_pat = vert_pat_(pheno);
    auto pheno_father = pheno_father_.col(pheno);
    auto pheno_mother = pheno_mother_.col(pheno);
    auto pheno_off = phenos_off.col(pheno);
    double norm =
        1.0 /
        std::sqrt((vert_pat * vert_pat) + ((1 - vert_pat) * (1 - vert_pat)));

    for (std::size_t pair = 0; pair < n_sex_; ++pair) {
      pheno_off(pair) = (vert_pat * pheno_father(pair)) +
                        ((1 - vert_pat) * pheno_mother(match[pair]));
      pheno_off(n_sex_ + pair) = pheno_off(pair);
    }

    phenos_off.col(pheno) *= std::sqrt(var_vert_(pheno)) * norm;
  }
}

inline void ScorePhenotypes::scoreTotal(State& state) {
  state.pheno()(Component::Total).noalias() =
      state.pheno()(Component::Genetic) +
      state.pheno()(Component::Environmental) +
      state.pheno()(Component::Vertical);
}

inline void ScorePhenotypes::operator()(State& state) {
  scoreGenetic(state);
  scoreEnvironmental(state);
  scoreVertical(state);
  scoreTotal(state);
}

}  // namespace knoop
