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

#include <knoop/core/utils.h>
#include <knoop/estimate/external_process.h>
#include <knoop/estimate/sample.h>
#include <knoop/io/parse.h>
#include <knoop/sample/proband.h>

#include <boost/process.hpp>
#include <utility>

namespace knoop {

namespace bp = boost::process;

template <ProbandType P>
class SampleEstimatorExternalStrategy : public SampleEstimatorStrategy<P> {
 public:
  SampleEstimatorExternalStrategy(
      const std::string& sample_name,
      const std::filesystem::path& sample_dir,
      const std::string& name,
      const std::string& cmd,
      std::vector<std::string> args,
      std::size_t n_rows,
      std::size_t n_cols = 1,
      std::optional<std::vector<std::string>> row_labels = std::nullopt,
      std::optional<std::vector<std::string>> col_labels = std::nullopt)
      : SampleEstimatorStrategy<P>(
            name,
            sample_name,
            sample_dir,
            std::move(row_labels.value_or(std::vector<std::string>{})),
            std::move(col_labels.value_or(std::vector<std::string>{})),
            n_rows,
            n_cols),
        cmd_(std::move(process::resolveExecutable(cmd))),
        args_(std::move(args)),
        bfile_(sample_dir / "data"),
        out_path_(sample_dir / std::format("results_{}", name)) {
    args_.push_back("--bfile");
    args_.push_back(bfile_.string());
    args_.push_back("--out");
    args_.push_back(out_path_.string());
  }

  void compute() override {
    runCommand();
    Eigen::MatrixXd result = parseFile<Eigen::MatrixXd>(out_path_);
    if (static_cast<std::size_t>(result.rows()) != this->numRows() ||
        static_cast<std::size_t>(result.cols()) != this->numCols())
      throw std::runtime_error(
          std::format(
              "{}: expected {}x{}, got {}x{}",
              cmd_.string(),
              this->numRows(),
              this->numCols(),
              result.rows(),
              result.cols()));
    this->data_ = std::move(result);
  }

 private:
  bp::filesystem::path cmd_;
  std::vector<std::string> args_;
  std::filesystem::path bfile_;
  std::filesystem::path out_path_;

  void runCommand() {
    process::ProcessResult result = process::runProcess(cmd_, args_);

    if (!result.output.empty()) {
      if (result.output.back() == '\n') result.output.pop_back();
      if (result.exit_code != 0)
        Log::error("{}", result.output);
      else
        Log::info("{}", result.output);
    }

    if (result.exit_code != 0)
      throw std::runtime_error(
          std::format(
              "{} exited with code {}, check log for detailed output",
              cmd_.string(),
              result.exit_code));
  }
};

template <ProbandType P>
inline SampleEstimator<P> sampleExternal(
    std::string name,
    std::string cmd,
    std::vector<std::string> args,
    std::size_t n_rows,
    std::size_t n_cols = 1,
    std::optional<std::vector<std::string>> row_labels = std::nullopt,
    std::optional<std::vector<std::string>> col_labels = std::nullopt) {
  return SampleEstimator<P>{
      .name = name,
      .fn = [name = std::move(name),
             cmd = std::move(cmd),
             args = std::move(args),
             n_rows,
             n_cols,
             row_labels = std::move(row_labels),
             col_labels = std::move(col_labels)](
                const Params& /*params*/,
                std::size_t /*n_probands*/,
                const std::filesystem::path& sample_dir) {
        return std::make_unique<SampleEstimatorExternalStrategy<P>>(
            sample_dir.filename().string(),
            sample_dir,
            name,
            cmd,
            args,
            n_rows,
            n_cols,
            row_labels,
            col_labels);
      }};
}

}  // namespace knoop
