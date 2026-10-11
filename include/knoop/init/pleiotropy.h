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

#include <knoop/core.h>

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace knoop {

// phenotype genetic component correlation optimiser
class OptimisePhenotypeArchitecture {
 public:
  explicit OptimisePhenotypeArchitecture(Params& params)
      : engine_(random::Xoshiro256pp::getInstance()),
        n_pheno_(params.pheno.n_pheno),
        n_loc_(params.geno.n_loc),
        n_locs_(params.pheno.n_locs),
        gen_cor_(params.pheno.gen_cor),
        h2_gen_(params.pheno.var_gen),
        pheno_effects_vec_(params.pheno.pheno_effects),
        pheno_loc_(params.pheno.pheno_loc),
        pheno_loc_complement_(n_pheno_),
        pheno_effects_(n_loc_, n_pheno_),
        pheno_fixed_(n_pheno_),
        diff_cur_(n_pheno_, n_pheno_) {
    gen_cov_ = h2_gen_.cwiseSqrt().asDiagonal() * gen_cor_ *
               h2_gen_.cwiseSqrt().asDiagonal();
    for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno)
      pheno_fixed_[pheno] = (n_locs_[pheno] == n_loc_);
  }

  double errFrobenius() const { return err_frob_opt_; }

  Eigen::MatrixXd expected() const { return expected_; }

  void operator()();

 private:
  boost::random::xoshiro256pp& engine_;

  const std::size_t n_pheno_;
  const std::size_t n_loc_;
  const std::vector<std::size_t>& n_locs_;
  const Eigen::MatrixXd& gen_cor_;
  const Eigen::VectorXd& h2_gen_;
  Eigen::MatrixXd gen_cov_;
  const std::vector<Eigen::VectorXd>& pheno_effects_vec_;
  std::vector<std::vector<std::size_t>>& pheno_loc_;
  std::vector<std::vector<std::size_t>> pheno_loc_complement_;
  Eigen::MatrixXd pheno_effects_;
  std::vector<bool> pheno_fixed_;

  std::size_t max_itr_ = 2e6;
  double tol_l2_ = 1e-15;
  double temp_init_ = 1.0;
  double temp_decay_ = 0.9999;
  std::size_t n_itr_;

  Eigen::MatrixXd expected_;
  Eigen::MatrixXd diff_cur_;
  Eigen::VectorXd u_cur_;
  double d_cur_;
  double temp_cur_;

  std::size_t pheno_cur_;
  std::size_t loc_ind_cur_;
  std::size_t loc_ind_prop_;
  std::size_t loc_cur_;
  std::size_t loc_prop_;

  double delta_cur_;
  double err_frob_;
  double err_frob_opt_;
  bool proposal_valid_;

  // generate the initial state
  void randomState();

  // compute the initial objective
  void computeInitObjective();

  // compute the expected correlation
  void computeExpected();

  // generate a proposal state (phenotype + switch) and compute update
  void proposeState();

  // metropolis accept step and update
  void updateState();
};

inline void OptimisePhenotypeArchitecture::randomState() {
  pheno_effects_.setZero();
  std::vector<std::size_t> iota(n_loc_);

  // NOLINTBEGIN(modernize-use-ranges)
  std::iota(iota.begin(), iota.end(), 0);
  // NOLINTEND(modernize-use-ranges)

  for (std::size_t pheno = 0; pheno < n_pheno_; ++pheno) {
    std::size_t n_locs = n_locs_[pheno];
    std::vector<std::size_t>& pheno_loc = pheno_loc_[pheno];
    std::vector<std::size_t>& pheno_loc_complement =
        pheno_loc_complement_[pheno];
    std::unordered_set<std::size_t> added;

    // random assignment using floyd's algorithm
    for (std::size_t loc = (n_loc_ - n_locs); loc < n_loc_; ++loc) {
      std::size_t gen = random::UniformInt::sample(engine_, 0, loc + 1);
      if (added.contains(gen)) gen = loc;
      added.insert(gen);
      pheno_loc[loc - (n_loc_ - n_locs)] = gen;
    }

    // fill the complement mask
    std::ranges::sort(pheno_loc);
    std::ranges::set_difference(
        iota, pheno_loc, std::back_inserter(pheno_loc_complement));

    // create the phenotype effect matrix wrt the assignment
    for (std::size_t loc = 0; loc < n_locs; ++loc)
      pheno_effects_(pheno_loc[loc], pheno) = pheno_effects_vec_[pheno][loc];
  }
}

inline void OptimisePhenotypeArchitecture::computeInitObjective() {
  diff_cur_ = pheno_effects_.transpose() * pheno_effects_ - gen_cov_;
  err_frob_ = diff_cur_.squaredNorm();
  err_frob_opt_ = err_frob_;
}

inline void OptimisePhenotypeArchitecture::computeExpected() {
  expected_ = pheno_effects_.transpose() * pheno_effects_;
}

inline void OptimisePhenotypeArchitecture::proposeState() {
  do {
    pheno_cur_ = random::UniformInt::sample(engine_, 0, n_pheno_);
  } while (pheno_fixed_[pheno_cur_]);

  loc_ind_cur_ = random::UniformInt::sample(engine_, 0, n_locs_[pheno_cur_]);
  loc_ind_prop_ =
      random::UniformInt::sample(engine_, 0, n_loc_ - n_locs_[pheno_cur_]);
  loc_cur_ = pheno_loc_[pheno_cur_][loc_ind_cur_];
  loc_prop_ = pheno_loc_complement_[pheno_cur_][loc_ind_prop_];

  double gamma_prop = pheno_effects_(loc_cur_, pheno_cur_);

  u_cur_ = gamma_prop *
           (pheno_effects_.row(loc_prop_) - pheno_effects_.row(loc_cur_))
               .transpose();

  d_cur_ = 2 * gamma_prop * gamma_prop;

  delta_cur_ = (2 * u_cur_.squaredNorm()) + (d_cur_ * d_cur_) +
               (4 * d_cur_ * u_cur_(pheno_cur_)) +
               (2 * u_cur_(pheno_cur_) * u_cur_(pheno_cur_)) +
               4 * diff_cur_.col(pheno_cur_).transpose() * u_cur_ +
               (2 * d_cur_ * diff_cur_(pheno_cur_, pheno_cur_));
}

inline void OptimisePhenotypeArchitecture::updateState() {
  double acc_prob = std::min(1.0, std::exp(-delta_cur_ / temp_cur_));
  double u = random::Uniform::sample(engine_);

  // accept
  if (u < acc_prob) {
    // update the locus masks and effect matrix
    std::swap(
        pheno_loc_[pheno_cur_][loc_ind_cur_],
        pheno_loc_complement_[pheno_cur_][loc_ind_prop_]);
    std::swap(
        pheno_effects_.col(pheno_cur_)(loc_cur_),
        pheno_effects_.col(pheno_cur_)(loc_prop_));

    // update the frobenius norm
    err_frob_ += delta_cur_;
    err_frob_opt_ = std::min(err_frob_, err_frob_opt_);

    // update the objective using rank-one update
    diff_cur_.col(pheno_cur_) += u_cur_;
    diff_cur_.row(pheno_cur_) += u_cur_.transpose();
    diff_cur_(pheno_cur_, pheno_cur_) += d_cur_;
  }
}

inline void OptimisePhenotypeArchitecture::operator()() {
  randomState();
  computeInitObjective();

  if (std::ranges::all_of(pheno_fixed_, std::identity{})) {
    Log::warning(
        "OptimisePhenotypeArchitecture::operator(): All phenotypes saturated "
        "in the genome; can not optimise phenotype "
        "architecture. Aborting.");
    return;
  }

  temp_cur_ = temp_init_;

  for (std::size_t itr = 0; itr < max_itr_; ++itr) {
    proposeState();
    updateState();
    temp_cur_ *= temp_decay_;

    if (err_frob_opt_ < tol_l2_) {
      n_itr_ = itr;
      break;
    }
  }

  // consider adding some tol_inf argument, decided against it for now
  Log::debug(
      "OptimisePhenotypeArchitecture::operator(): finished with error {}",
      errFrobenius());
}

}  // namespace knoop
