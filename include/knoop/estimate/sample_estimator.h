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
#include <knoop/sample/proband.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace knoop {

struct Params;

template <ProbandType P>
class SampleEstimatorStrategy;

struct SampleEstimatorSpec {
  std::string name;
  std::string type;
  std::optional<std::string> exec;
  std::vector<std::string> exec_args;
  std::optional<std::size_t> n_rows;
  std::optional<std::size_t> n_cols;
  std::optional<std::vector<std::string>> row_names;
  std::optional<std::vector<std::string>> col_names;
  std::vector<std::string> params;
};

template <ProbandType P>
struct SampleEstimatorFor {
  std::string name;
  std::function<std::unique_ptr<SampleEstimatorStrategy<P>>(
      const Params&, std::size_t, const std::filesystem::path&)>
      fn;

  std::unique_ptr<SampleEstimatorStrategy<P>> operator()(
      const Params& params,
      std::size_t n_probands,
      const std::filesystem::path& sample_dir) const {
    assert(
        omp_pause_resource_all(omp_pause_soft) != 0 &&
        "Could not perform soft pause of OMP threads");

    return fn(params, n_probands, sample_dir);
  }
};

/**
 * \brief Defers specialisation of class to a proband type to allow reuse of the
 * same sample estimator declaration across many study types
 */
class SampleEstimator {
 private:
  /**
   * Has three function fields, each of which constructs the specialised variant
   * which can be attached to a sample
   */
  std::function<SampleEstimatorFor<ProbandType::Self>()> self_;
  std::function<SampleEstimatorFor<ProbandType::Family>()> family_;
  std::function<SampleEstimatorFor<ProbandType::Mate>()> mate_;

 public:
  /**
   * Take a recipe for how the estimator is usually declared and store the
   * specialisation calls in each of the respective member functions self_,
   * family_, and mate_.
   */
  template <typename Specialiser>
  explicit SampleEstimator(Specialiser spec)
      : self_(
            [spec]() { return spec.template operator()<ProbandType::Self>(); }),
        family_([spec]() {
          return spec.template operator()<ProbandType::Family>();
        }),
        mate_([spec]() {
          return spec.template operator()<ProbandType::Mate>();
        }) {}

  template <ProbandType P>
  SampleEstimatorFor<P> build() const {
    if constexpr (P == ProbandType::Self) return self_();
    if constexpr (P == ProbandType::Family) return family_();
    if constexpr (P == ProbandType::Mate) return mate_();
    std::unreachable();
  }
};

template <ProbandType P>
using SampleEstimators = std::vector<SampleEstimatorFor<P>>;

}  // namespace knoop
