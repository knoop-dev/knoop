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
struct SampleEstimator {
  std::string name;
  std::function<std::unique_ptr<SampleEstimatorStrategy<P>>(
      const Params&, std::size_t, const std::filesystem::path&)>
      fn;

  std::unique_ptr<SampleEstimatorStrategy<P>> operator()(
      const Params& params,
      std::size_t n_probands,
      const std::filesystem::path& sample_dir) const {
    return fn(params, n_probands, sample_dir);
  }
};

template <ProbandType P>
using SampleEstimators = std::vector<SampleEstimator<P>>;

}  // namespace knoop
