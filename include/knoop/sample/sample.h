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

#include <knoop/data/component.h>
#include <knoop/estimate/sample_estimator.h>
#include <knoop/io/parse.h>
#include <knoop/sample/proband.h>
#include <knoop/sample/selection.h>

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace knoop {

// user-facing specification struct
template <ProbandType P>
struct SampleFor {
  // selection parameters — which data, from whom, and what to do
  std::string name;
  std::size_t n_probands;
  std::optional<std::vector<std::string>> phenotypes_on;
  std::optional<std::vector<Component>> phenotypes_components;
  ProbandMemberEnum<P> members_of = kProbandOfDefault<P>;
  ProbandMemberEnum<P> members_include = kProbandIncludeDefault<P>;
  Aggregator aggregator = kProbandAggregatorDefault<P>;

  // weighting method — probability of selecting based on proband aggregate
  Weight weight = Weight{.weight_fn = weightUniform()};

  // estimators are declared here
  std::vector<SampleEstimator> estimators;

  // defined out-of-line in sample/sampling.h, where build_sample_estimator's
  // full implementation (and everything it depends on) is available
  void attachEstimator(const SampleEstimator& estimator);
};

using Sample = std::variant<
    SampleFor<ProbandType::Self>,
    SampleFor<ProbandType::Mate>,
    SampleFor<ProbandType::Family>>;

}  // namespace knoop
