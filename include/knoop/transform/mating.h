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
#include <knoop/data/matching.h>
#include <knoop/io/hdf5.h>
#include <knoop/io/table.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

namespace knoop {

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class RandomMating {
 public:
  explicit RandomMating(const Params& params)
      : eng_(random::Xoshiro256pp::getInstance()),
        n_sex_(params.global.n_ind / 2),
        match_cur_(n_sex_) {}

  void operator()(State& state);

 private:
  boost::random::xoshiro256pp& eng_;
  std::size_t n_sex_;
  Matching match_cur_;

  void randomiseMatching();
};

inline void RandomMating::randomiseMatching() {
  // NOLINTBEGIN(modernize-use-ranges)
  std::iota(match_cur_.begin(), match_cur_.end(), 0);
  // NOLINTEND(modernize-use-ranges)

  for (std::size_t el = 0; el < n_sex_; ++el) {
    std::size_t le = random::UniformInt::sample(eng_, el, n_sex_);
    std::swap(match_cur_[el], match_cur_[le]);
  }
}

inline void RandomMating::operator()(State& state) {
  randomiseMatching();
  state.matching() = match_cur_;
  for (std::size_t ind = 0; ind < n_sex_; ++ind)
    state.invMatching()[state.matching()[ind]] = ind;
}

class AssortativeMating {
 public:
  AssortativeMating(const AssortativeMating&) = delete;
  AssortativeMating(AssortativeMating&&) = delete;
  AssortativeMating& operator=(const AssortativeMating&) = delete;
  AssortativeMating& operator=(AssortativeMating&&) = delete;
  explicit AssortativeMating(const Params& params)
      : n_sex_(params.global.n_ind / 2),
        n_pheno_(params.pheno.n_pheno),
        std_male_(params.global.n_ind, params.pheno.n_pheno),
        std_female_(params.global.n_ind, params.pheno.n_pheno),
        mate_cor_(params.mate.mate_cor),
        objective_(params.mate.mate_cor),
        exact_(params.mate.exact),
        engine_(random::Xoshiro256pp::getInstance()),
        match_cur_(n_sex_),
        max_itr_(
            params.mate.max_itr == 0
                ? std::max<std::size_t>(10 * params.global.n_ind, 100000)
                : params.mate.max_itr),
        temp_init_(params.mate.temp_init),
        temp_decay_(params.mate.temp_decay),
        tol_inf_(params.mate.tol / std::sqrt(n_sex_)),
        stall_window_(n_sex_),
        log_interval_(std::max<std::size_t>(max_itr_ / 1000, 1)) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mate_cor_);

    if (svd.singularValues()(0) > 1)
      throw std::invalid_argument(
          "AssortativeMating: invalid spouse correlation matrix specified; "
          "singular value(s) exceed one");
  };

  void operator()(State& state);

  // optimal matching
  Matching matching() const { return match_cur_; };

  // best l2 error achieved by solver
  double l2Error() const { return err_l2_opt_; }

  // max abs error of the matching returned by the solver
  double lInfError() const { return err_linf_l2opt_; }

  // smallest max abs error visited during the run
  double lInfErrorOpt() const { return err_linf_opt_; }

  // record per-iteration annealer diagnostics; off by default since it costs
  // one table row per iteration
  void enableDiagnostics() { diagnostics_ = true; }

  // write the collected diagnostics; no-op if diagnostics were never enabled
  void writeDiagnostics(const std::filesystem::path& path) const {
    if (diagnostics_) writeTable(diagnostics_tbl_, path, ',');
  }

  Eigen::MatrixXd objective() const { return objective_; }

  // best correlation structure achieved by solver
  Eigen::MatrixXd realisedCor() const { return cor_; };

  // number of iterations performed by solver
  std::size_t numIterations() const { return n_itr_; };

 private:
  std::size_t n_sex_;
  std::size_t n_pheno_;
  RowMatrixXd std_male_;
  RowMatrixXd std_female_;
  Eigen::MatrixXd mate_cor_;
  Eigen::MatrixXd objective_;
  bool exact_;

  boost::random::xoshiro256pp& engine_;

  std::size_t id_swap_l_;
  std::size_t id_swap_r_;
  Matching match_cur_;
  Eigen::MatrixXd ell_cur_;
  Eigen::MatrixXd ell_opt_;
  double alpha_cur_;
  double err_l2_cur_;
  double err_l2_opt_;
  double err_l2_window_;
  double err_linf_cur_;
  double err_linf_opt_;
  double err_linf_l2opt_;
  bool tol_met_ = false;

  Eigen::VectorXd diff_male_;
  Eigen::VectorXd diff_female_;
  Eigen::VectorXd ell_diff_female_;

  std::vector<std::pair<std::size_t, std::size_t>> undo_;

  Eigen::MatrixXd cor_;
  std::size_t n_itr_;
  std::size_t max_itr_;

  double temp_init_;
  double temp_decay_;
  double temp_cur_;
  double tol_inf_;
  std::size_t stall_window_;
  double stall_rel_gain_ = 0.01;

  std::size_t log_interval_;

  bool accept_cur_ = false;
  bool diagnostics_ = false;
  std::size_t n_accept_interval_ = 0;
  std::size_t itr_last_log_ = 0;

  Table<
      Column<"iteration", std::size_t>,
      Column<"prop_energy", double>,
      Column<"prop_accepted", bool>,
      Column<"accept_rate", double>,
      Column<"l2_error", double>,
      Column<"linf_error", double>,
      Column<"linf_error_opt", double>,
      Column<"temperature", double>>
      diagnostics_tbl_;

  // append one diagnostics row summarising the interval ending at itr
  void recordDiagnostics(std::size_t itr);

  // generate a random matrix centred around mate_cor_ to mimic natural
  // variation attributable to finite population size
  void generateObjective();

  Matching invMatching();

  // shuffle the internal matching using Fisher-Yates
  void randomiseMatching();

  // compute the cross-correlation matrix for the initial state
  void computeCor();
  void proposeState();

  // accept or reject the proposed step
  void updateState();

  // roll match_cur_ back to the recorded optimum
  void rewindToOptimumMatching();
};

inline void AssortativeMating::generateObjective() {
  double n_inv = 1.0 / n_sex_;

  Eigen::MatrixXd male_cor = n_inv * std_male_.transpose() * std_male_;
  Eigen::MatrixXd female_cor = n_inv * std_female_.transpose() * std_female_;
  Eigen::LLT<Eigen::MatrixXd> llt_male(male_cor);

  if (llt_male.info() != Eigen::Success)
    throw std::runtime_error(
        "AssortativeMating: male trait correlation matrix is singular");

  Eigen::LLT<Eigen::MatrixXd> llt_col(
      female_cor - (mate_cor_.transpose() * llt_male.solve(mate_cor_)));

  if (llt_col.info() != Eigen::Success)
    throw std::runtime_error(
        "AssortativeMating: unable to perform random draw of mate correlation "
        "matrix due to male and trait correlation matrices");

  const Eigen::MatrixXd l_row = llt_male.matrixL();
  const Eigen::MatrixXd l_col = llt_col.matrixL();

  constexpr std::size_t kNumAttempts = 8;

  Eigen::MatrixXd rand(n_pheno_, n_pheno_);
  for (std::size_t attempt = 0; attempt < kNumAttempts; ++attempt) {
    random::Normal::sample(engine_, rand.data(), n_pheno_ * n_pheno_);
    objective_ =
        mate_cor_ + std::sqrt(n_inv) * l_row * rand * l_col.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_obj(objective_);

    if (svd_obj.singularValues()(0) < 1) return;
  }

  Log::warning(
      "AssortativeMating: unable to draw feasible mate correlation matrix "
      "after 8 tries; falling back to mate_cor");

  objective_ = mate_cor_;
}

inline void AssortativeMating::randomiseMatching() {
  // NOLINTBEGIN(modernize-use-ranges)
  std::iota(match_cur_.begin(), match_cur_.end(), 0);
  // NOLINTEND(modernize-use-ranges)
  for (std::size_t el = 0; el < n_sex_; ++el) {
    std::size_t le = random::UniformInt::sample(engine_, el, n_sex_);
    std::swap(match_cur_[el], match_cur_[le]);
  }
}

inline Matching AssortativeMating::invMatching() {
  Matching inv_match(n_sex_);
  for (std::size_t ind = 0; ind < n_sex_; ++ind)
    inv_match[match_cur_[ind]] = ind;
  return inv_match;
}

inline void AssortativeMating::computeCor() {
  cor_ = Eigen::MatrixXd::Zero(n_pheno_, n_pheno_);
  for (std::size_t pair = 0; pair < n_sex_; ++pair)
    cor_.noalias() +=
        std_male_.row(pair).transpose() * std_female_.row(match_cur_[pair]);
  cor_ /= static_cast<double>(n_sex_);
}

inline void AssortativeMating::proposeState() {
  id_swap_l_ = random::UniformInt::sample(engine_, 0, n_sex_);
  id_swap_r_ = random::UniformInt::sample(engine_, 0, n_sex_);
  while (id_swap_l_ == id_swap_r_)
    id_swap_r_ = random::UniformInt::sample(engine_, 0, n_sex_);

  diff_male_ =
      (std_male_.row(id_swap_l_) - std_male_.row(id_swap_r_)).transpose();
  diff_female_ = (std_female_.row(match_cur_[id_swap_r_]) -
                  std_female_.row(match_cur_[id_swap_l_]))
                     .transpose();

  double inv_n = 1.0 / static_cast<double>(n_sex_);
  ell_diff_female_.noalias() = ell_cur_ * diff_female_;

  alpha_cur_ =
      (diff_male_.squaredNorm() * diff_female_.squaredNorm() * inv_n * inv_n) +
      (2.0 * inv_n * diff_male_.dot(ell_diff_female_));
}

inline void AssortativeMating::updateState() {
  // compute acceptance probability
  double acc_prob = std::min(1.0, std::exp(-alpha_cur_ / temp_cur_));
  double u = random::Uniform::sample(engine_);

  // accept / reject
  accept_cur_ = u < acc_prob;
  n_accept_interval_ += static_cast<std::size_t>(accept_cur_);

  if (accept_cur_) {
    std::swap(match_cur_[id_swap_l_], match_cur_[id_swap_r_]);
    ell_cur_.noalias() += (1.0 / static_cast<double>(n_sex_)) * diff_male_ *
                          diff_female_.transpose();

    undo_.emplace_back(id_swap_l_, id_swap_r_);

    err_l2_cur_ = ell_cur_.norm();
    err_linf_cur_ = ell_cur_.array().abs().maxCoeff();
    err_linf_opt_ = std::min(err_linf_opt_, err_linf_cur_);

    if (err_l2_cur_ < err_l2_opt_) {
      err_l2_opt_ = err_l2_cur_;
      err_linf_l2opt_ = err_linf_cur_;
      undo_.clear();
    } else if (undo_.size() > n_sex_ / 8) {
      // the undo log has grown unreasonably large; rewind to an earlier
      // optimum and continue from the best
      rewindToOptimumMatching();
      ell_cur_ = ell_opt_;
      err_l2_cur_ = err_l2_opt_;
      err_linf_cur_ = err_linf_l2opt_;
    }
  }
}

inline void AssortativeMating::recordDiagnostics(std::size_t itr) {
  std::size_t n_prop = std::max<std::size_t>(itr - itr_last_log_, 1);

  diagnostics_tbl_.insertRow(
      {itr,
       alpha_cur_,
       accept_cur_,
       static_cast<double>(n_accept_interval_) / static_cast<double>(n_prop),
       err_l2_cur_,
       err_linf_cur_,
       err_linf_opt_,
       temp_cur_});

  n_accept_interval_ = 0;
  itr_last_log_ = itr;
}

inline void AssortativeMating::rewindToOptimumMatching() {
  for (const auto& itr : std::ranges::reverse_view(undo_))
    std::swap(match_cur_[itr.first], match_cur_[itr.second]);
  undo_.clear();
}

inline void AssortativeMating::operator()(State& state) {
  randomiseMatching();

  utils::standardise(state.pheno().male(), std_male_);
  utils::standardise(state.pheno().female(), std_female_);

  if (!exact_) generateObjective();

  computeCor();

  // initialise values for this run
  temp_cur_ = temp_init_;
  ell_cur_ = cor_ - objective_;
  ell_opt_ = ell_cur_;
  err_l2_cur_ = ell_cur_.norm();
  err_l2_opt_ = err_l2_cur_;
  err_linf_cur_ = ell_cur_.array().abs().maxCoeff();
  err_linf_opt_ = err_linf_cur_;
  err_linf_l2opt_ = err_linf_cur_;
  tol_met_ = false;

  undo_.clear();

  n_accept_interval_ = 0;
  itr_last_log_ = 0;

  err_l2_window_ = err_l2_opt_;
  std::size_t next_check = stall_window_;
  bool stalled = false;
  n_itr_ = max_itr_;  // default if the loop runs to completion unbroken

  // annealing routine
  for (std::size_t itr = 0; itr < max_itr_; ++itr) {
    proposeState();
    updateState();

    temp_cur_ *= temp_decay_;

    if (itr % log_interval_ == 0) {
      if (diagnostics_) recordDiagnostics(itr);
      Log::debug(
          "AssortativeMating: iteration {}, error l2 {:.3g} (optimal "
          "{:.3g}), error linfty {:.3g} (optimal {:.3g}), temperature "
          "{:.3g}",
          itr,
          err_l2_cur_,
          err_l2_opt_,
          err_linf_cur_,
          err_linf_opt_,
          temp_cur_);
    }

    // tested on the current state, since that is the matching we return when
    // the tolerance is met
    if (err_linf_cur_ < tol_inf_) {
      n_itr_ = itr + 1;
      tol_met_ = true;
      break;
    }

    if (itr >= next_check) {
      if (err_l2_opt_ > err_l2_window_ * (1.0 - stall_rel_gain_)) {
        n_itr_ = itr + 1;
        stalled = true;
        break;
      }
      err_l2_window_ = err_l2_opt_;
      next_check *= 2;
    }
  }

  if (lInfError() > tol_inf_) {
    if (stalled) {
      Log::warning(
          "AssortativeMating: search stalled after {} iterations with no "
          "further improvement (max abs error: {:.3g}, l2 error: {:.3g}, "
          "tolerance: {:.3g}). The target may be unreachable at this n, or "
          "loosen --tol-inf.",
          n_itr_,
          lInfError(),
          l2Error(),
          tol_inf_);
    } else {
      Log::warning(
          "AssortativeMating: mate correlation error exceeds tolerance "
          "after {} iterations (max abs error: {:.3g}, l2 error: {:.3g}, "
          "tolerance: {:.3g}). Consider raising --max-itr or loosening "
          "--tol-inf.",
          n_itr_,
          lInfError(),
          l2Error(),
          tol_inf_);
    }
  }

  // on the tolerance break match_cur_ already is the state we want; otherwise
  // materialise the l2 optimum from the undo log
  if (tol_met_) {
    err_linf_l2opt_ = err_linf_cur_;
    err_l2_opt_ = err_l2_cur_;
    cor_ = ell_cur_ + objective_;
  } else {
    rewindToOptimumMatching();
    cor_ = ell_opt_ + objective_;
  }

  state.matching() = match_cur_;
  state.invMatching() = invMatching();
}

}  // namespace knoop
