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

#include <knoop/core/error.h>
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
      : n_ind_(params.numIndividuals()),
        n_sex_(n_ind_ / 2),
        n_pheno_(params.pheno.size()),
        phenotypes_(params.pheno),
        tile_size_(
            (n_pheno_ == 0) ? 0
                            : std::ranges::max(params.pheno.numCausalLoci())),
        env_chol_(params.pheno.environComponentCorrelation().llt().matrixL()),
        pheno_father_(n_sex_, n_pheno_),
        pheno_mother_(n_sex_, n_pheno_) {}

  void operator()(State& state);

 private:
  const std::size_t n_ind_;
  const std::size_t n_sex_;
  const std::size_t n_pheno_;

  const PhenotypeParams& phenotypes_;
  const std::size_t tile_size_;

  static constexpr std::size_t kIndTile = 512;
  Eigen::MatrixXd env_chol_;
  Eigen::MatrixXd pheno_father_;
  Eigen::MatrixXd pheno_mother_;

  void scoreGenetic(State& state);
  void scoreEnvironmental(State& state);
  void scoreIndirect(State& state);
  static void scoreTotal(State& state);
};

inline void ScorePhenotypes::scoreGenetic(State& state) {
  assert(
      state.geno().view() == BufferLayout::LocusMajor &&
      "Locus-major layout is required to score phenotypes");

  auto& genotype_buffer = state.geno();
  auto genetic_buffer = state.pheno()(Component::Genetic);

#pragma omp parallel
  {
    Eigen::MatrixXd gen_tile(kIndTile, tile_size_);

#pragma omp for collapse(2) schedule(static)
    for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno) {
      for (std::size_t ind_tile = 0; ind_tile < n_ind_; ind_tile += kIndTile) {
        std::size_t tile_size = std::min(kIndTile, n_ind_ - ind_tile);

        // decompress the standardised genotype matrix in manageable tiles
        genotype_buffer.decompress(
            ind_tile,
            ind_tile + tile_size,
            phenotypes_.causalLoci(pheno),
            gen_tile,
            true,
            true);

        // update the genetic component buffer in-place
        genetic_buffer.col(pheno).segment(ind_tile, tile_size).noalias() =
            gen_tile.topRows(tile_size).leftCols(
                phenotypes_.numCausalLoci(pheno)) *
            phenotypes_.effects(pheno);
      }
    }
  }

  if (!genetic_buffer.allFinite())
    throw KnoopNumericalError(
        "Infinite or NaN values in phenotype genetic component buffer");
}

inline void ScorePhenotypes::scoreEnvironmental(State& state) {
  auto environmental_buffer = state.pheno()(Component::Environmental);

  knoop::random::Normal::sample(
      random::Xoshiro256pp::getInstance(),
      environmental_buffer.data(),
      n_ind_ * n_pheno_);

  environmental_buffer =
      environmental_buffer * env_chol_.transpose() *
      phenotypes_.varEnvironmental().cwiseSqrt().asDiagonal();

  if (!environmental_buffer.allFinite())
    throw KnoopNumericalError(
        "Infinite or NaN values in phenotype genetic component buffer");
}

inline void ScorePhenotypes::scoreIndirect(State& state) {
  if (phenotypes_.varIndirect().isZero(0)) return;

  utils::standardise(
      state.pheno(Generation::Parents).male(Component::Total), pheno_father_);
  utils::standardise(
      state.pheno(Generation::Parents).female(Component::Total), pheno_mother_);

  auto indirect_offspring =
      state.pheno(Generation::Current)(Component::Indirect);
  auto& match = state.matching(Generation::Parents);

  // score siblings
#pragma omp for schedule(static)
  for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno) {
    double indirect_pat = phenotypes_.propIndirectPaternal(pheno);
    auto pheno_father = pheno_father_.col(pheno);
    auto pheno_mother = pheno_mother_.col(pheno);
    auto pheno_off = indirect_offspring.col(pheno);
    double norm = 1.0 / std::sqrt(
                            (indirect_pat * indirect_pat) +
                            ((1 - indirect_pat) * (1 - indirect_pat)));

    for (std::size_t pair = 0; pair < n_sex_; ++pair) {
      pheno_off(pair) = (indirect_pat * pheno_father(pair)) +
                        ((1 - indirect_pat) * pheno_mother(match[pair]));
      pheno_off(n_sex_ + pair) = pheno_off(pair);
    }

    indirect_offspring.col(pheno) *=
        std::sqrt(phenotypes_.varIndirect(pheno)) * norm;
  }

  if (!indirect_offspring.allFinite())
    throw KnoopNumericalError(
        "Infinite or NaN values in phenotype genetic component buffer");
}

inline void ScorePhenotypes::scoreTotal(State& state) {
  state.pheno()(Component::Total).noalias() =
      state.pheno()(Component::Genetic) +
      state.pheno()(Component::Environmental) +
      state.pheno()(Component::Indirect);
}

inline void ScorePhenotypes::operator()(State& state) {
  scoreGenetic(state);
  scoreEnvironmental(state);
  scoreIndirect(state);
  scoreTotal(state);

  if (!state.pheno()(Component::Total).allFinite())
    throw KnoopNumericalError("Infinite or NaN values in phenotype buffer");
}

}  // namespace knoop
