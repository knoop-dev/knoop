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
#include <knoop/estimate/genome_estimators.h>
#include <knoop/estimate/mating_estimators.h>
#include <knoop/estimate/pedigree_estimators.h>
#include <knoop/estimate/phenome_estimators.h>

#include <memory>
#include <vector>

namespace knoop {

class ComputePopulationEstimates {
 public:
  explicit ComputePopulationEstimates(const Params& params) {
    for (const auto& factory : params.estimate.population_estimators)
      estimators_.emplace_back(factory(params));
  }

  void operator()(const State& state);

 private:
  std::vector<std::unique_ptr<PopulationEstimatorStrategy>> estimators_;
};

inline void ComputePopulationEstimates::operator()(const State& state) {
  for (auto& estimator : estimators_) (*estimator)(state);
}

}  // namespace knoop
