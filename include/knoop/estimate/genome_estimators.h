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
#include <knoop/estimate/estimator.h>
#include <knoop/estimate/strategy.h>

#include <Eigen/Dense>

namespace knoop {

namespace details {

class EstimatorGenotypeFreqStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorGenotypeFreqStrategy(const Params& params)
      : PopulationEstimatorStrategy(
            "geno_freq", {"00", "10/01", "11"}, {}, 3, params.geno.n_loc) {}

  void compute(const State& state) override {
    data_ = state.geno().genotypeFreq();
  }
};

class EstimatorGenotypeMeanStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorGenotypeMeanStrategy(const Params& params)
      : PopulationEstimatorStrategy("geno_mean", {}, {}, params.geno.n_loc) {}

  void compute(const State& state) override {
    data_ = state.geno().locusMean();
  }
};

class EstimatorGenotypeVarStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorGenotypeVarStrategy(const Params& params)
      : PopulationEstimatorStrategy("geno_var", {}, {}, params.geno.n_loc) {}

  void compute(const State& state) override { data_ = state.geno().locusVar(); }
};

class EstimatorAlleleFreqStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorAlleleFreqStrategy(const Params& params)
      : PopulationEstimatorStrategy("allele_freq", {}, {}, params.geno.n_loc) {}

  void compute(const State& state) override {
    data_ = state.geno().alleleFreq();
  }
};

class EstimatorGenotypeCovStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorGenotypeCovStrategy(const Params& params)
      : PopulationEstimatorStrategy(
            "geno_cov", {}, {}, params.geno.n_loc, params.geno.n_loc) {
    n_loc_ = params.geno.n_loc;
  }

  void compute(const State& state) override {
    const GenotypeBuffer& geno = state.geno();
    std::size_t n_words = geno.words();
    std::size_t n_ind = geno.individuals();

    for (std::size_t loc1 = 0; loc1 < n_loc_; ++loc1) {
      // haplotypes of the first locus
      const uint64_t* h01 = geno.paternal().rowptr(loc1);
      const uint64_t* h11 = geno.maternal().rowptr(loc1);
      for (std::size_t loc2 = loc1; loc2 < n_loc_; ++loc2) {
        // haplotypes of the second locus
        const uint64_t* h02 = geno.paternal().rowptr(loc2);
        const uint64_t* h12 = geno.maternal().rowptr(loc2);
        std::size_t acc = 0;

        for (std::size_t word = 0; word < n_words; ++word)
          if (word == n_words - 1 && (n_ind % 64)) {
            std::uint64_t mask = (1ULL << (n_ind % 64)) - 1ULL;
            acc += __builtin_popcountll(h01[word] & h02[word] & mask) +
                   __builtin_popcountll(h01[word] & h12[word] & mask) +
                   __builtin_popcountll(h11[word] & h02[word] & mask) +
                   __builtin_popcountll(h11[word] & h12[word] & mask);
          } else {
            acc += __builtin_popcountll(h01[word] & h02[word]) +
                   __builtin_popcountll(h01[word] & h12[word]) +
                   __builtin_popcountll(h11[word] & h02[word]) +
                   __builtin_popcountll(h11[word] & h12[word]);
          }

        data_(loc1, loc2) = ((1.0 / n_ind) * acc) -
                            (geno.locusMean(loc1) * geno.locusMean(loc2));
      }
    }

    data_.triangularView<Eigen::Lower>() =
        data_.transpose().triangularView<Eigen::Lower>();
  }

 private:
  std::size_t n_loc_;
};

class EstimatorGenotypeCorStrategy : public PopulationEstimatorStrategy {
 public:
  explicit EstimatorGenotypeCorStrategy(const Params& params)
      : PopulationEstimatorStrategy(
            "geno_cor", {}, {}, params.geno.n_loc, params.geno.n_loc) {
    n_loc_ = params.geno.n_loc;
  }

  void compute(const State& state) override {
    const GenotypeBuffer& geno = state.geno();
    std::size_t n_words = geno.words();
    std::size_t n_ind = geno.individuals();

    for (std::size_t loc1 = 0; loc1 < n_loc_; ++loc1) {
      const uint64_t* h01 = geno.paternal().rowptr(loc1);
      const uint64_t* h11 = geno.maternal().rowptr(loc1);
      for (std::size_t loc2 = loc1; loc2 < n_loc_; ++loc2) {
        const uint64_t* h02 = geno.paternal().rowptr(loc2);
        const uint64_t* h12 = geno.maternal().rowptr(loc2);
        std::size_t acc = 0;

        for (std::size_t word = 0; word < n_words; ++word) {
          if (word == n_words - 1 && (n_ind % 64)) {
            std::uint64_t mask = (1ULL << (n_ind % 64)) - 1ULL;
            acc += __builtin_popcountll(h01[word] & h02[word] & mask) +
                   __builtin_popcountll(h01[word] & h12[word] & mask) +
                   __builtin_popcountll(h11[word] & h02[word] & mask) +
                   __builtin_popcountll(h11[word] & h12[word] & mask);
          } else {
            acc += __builtin_popcountll(h01[word] & h02[word]) +
                   __builtin_popcountll(h01[word] & h12[word]) +
                   __builtin_popcountll(h11[word] & h02[word]) +
                   __builtin_popcountll(h11[word] & h12[word]);
          }
        }

        double cov = ((1.0 / n_ind) * acc) -
                     (geno.locusMean(loc1) * geno.locusMean(loc2));
        double denom = std::sqrt(geno.locusVar(loc1) * geno.locusVar(loc2));
        data_(loc1, loc2) = (denom > 0.0) ? cov / denom : 0.0;
      }
    }

    data_.triangularView<Eigen::Lower>() =
        data_.transpose().triangularView<Eigen::Lower>();
  }

 private:
  std::size_t n_loc_;
};

}  // namespace details

inline PopulationEstimator populationGenotypeFreq() {
  return PopulationEstimator{
      .name = "genotype-freq", .fn = [](const Params& params) {
        return std::make_unique<details::EstimatorGenotypeFreqStrategy>(params);
      }};
}

inline PopulationEstimator populationGenotypeCov() {
  return PopulationEstimator{
      .name = "genotype-cov", .fn = [](const Params& params) {
        return std::make_unique<details::EstimatorGenotypeCovStrategy>(params);
      }};
}

inline PopulationEstimator populationGenotypeMean() {
  return PopulationEstimator{
      .name = "genotype-mean", .fn = [](const Params& params) {
        return std::make_unique<details::EstimatorGenotypeMeanStrategy>(params);
      }};
}

inline PopulationEstimator populationGenotypeVar() {
  return PopulationEstimator{
      .name = "genotype-var", .fn = [](const Params& params) {
        return std::make_unique<details::EstimatorGenotypeVarStrategy>(params);
      }};
}

inline PopulationEstimator populationAlleleFreq() {
  return PopulationEstimator{
      .name = "allele-freq", .fn = [](const Params& params) {
        return std::make_unique<details::EstimatorAlleleFreqStrategy>(params);
      }};
}

inline PopulationEstimator populationGenotypeCor() {
  return PopulationEstimator{
      .name = "genotype-cor", .fn = [](const Params& params) {
        return std::make_unique<details::EstimatorGenotypeCorStrategy>(params);
      }};
}

}  // namespace knoop
