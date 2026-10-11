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

#include <knoop/core/log.h>
#include <knoop/core/params.h>
#include <knoop/core/utils.h>
#include <knoop/estimate/external_process.h>
#include <knoop/estimate/sample.h>
#include <knoop/io/hdf5.h>
#include <knoop/io/table.h>
#include <knoop/sample/proband.h>

#include <Eigen/Dense>
#include <boost/process.hpp>
#include <filesystem>
#include <limits>
#include <utility>

namespace knoop {

namespace bp = boost::process;

template <ProbandType P>
class SampleEstimatorGREMLStrategy : public SampleEstimatorStrategy<P> {
 public:
  SampleEstimatorGREMLStrategy(
      const Params& params,
      std::string sample_name,
      std::filesystem::path sample_dir,
      std::string name = "greml")
      : SampleEstimatorStrategy<P>(
            std::move(name),
            std::move(sample_name),
            sample_dir,
            params.pheno.names,
            {"V(G)", "V(E)", "V(G)/[V(G) + V(E)]"},
            params.pheno.n_pheno,
            3),
        gcta64_(std::move(process::resolveExecutable("gcta64"))),
        n_pheno_(params.pheno.n_pheno),
        pheno_names_(params.pheno.names) {}

  void compute() override {
    if (!std::filesystem::exists(this->sample_dir_ / "grm.grm.bin")) {
      process::runProcess(
          gcta64_,
          {"--bfile",
           (this->sample_dir_ / "data").string(),
           "--make-grm",
           "--out",
           (this->sample_dir_ / "grm").string(),
           "--thread-num",
           "1"});
    }

    for (std::size_t p = 0; p < n_pheno_; ++p) {
      auto pfix = this->sample_dir_ / std::format("greml_{}", p);

      process::runProcess(
          gcta64_,
          {"--grm",
           (this->sample_dir_ / "grm").string(),
           "--pheno",
           (this->sample_dir_ / "data.pheno").string(),
           "--mpheno",
           std::format("{}", p + 1),
           "--reml",
           "--out",
           pfix.string(),
           "--thread-num",
           "1"});

      this->data_.row(p) = parseHsq(pfix.string() + ".hsq", p);
    }
  }

 private:
  bp::filesystem::path gcta64_;
  std::size_t n_pheno_;
  std::vector<std::string> pheno_names_;

  Eigen::RowVector3d parseHsq(const std::string& path, std::size_t p) {
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    Eigen::RowVector3d result{NaN, NaN, NaN};

    if (!std::filesystem::exists(path)) {
      Log::error(
          "{}: gcta64 --reml exited successfully but {} was never "
          "written, for phenotype '{}'",
          this->name_,
          path,
          pheno_names_[p]);
      return result;
    }

    Table<Column<"Source", std::string>, Column<"Variance", double>> hsq;
    hsq.readFile(path, '\t');

    bool any_matched = false;
    for (std::size_t r = 0; r < hsq.numRows(); ++r) {
      auto [source, variance] = hsq.row(r);
      if (source == "V(G)") {
        result(0) = variance;
        any_matched = true;
      } else if (source == "V(e)") {
        result(1) = variance;
        any_matched = true;
      } else if (source == "V(G)/Vp") {
        result(2) = variance;
        any_matched = true;
      }
    }
    if (!any_matched)
      Log::error(
          "{}: {} parsed but none of V(G)/V(e)/V(G)/Vp were found for "
          "phenotype '{}' — check gcta64's .hsq format hasn't changed",
          this->name_,
          path,
          pheno_names_[p]);
    return result;
  }
};

template <ProbandType P>
inline SampleEstimator<P> sampleGREML(std::string name = "greml") {
  return SampleEstimator<P>{
      .name = name,
      .fn = [name = std::move(name)](
                const Params& params,
                std::size_t /*n_probands*/,
                const std::filesystem::path& sample_dir) {
        return std::make_unique<SampleEstimatorGREMLStrategy<P>>(
            params, sample_dir.filename().string(), sample_dir, name);
      }};
}

}  // namespace knoop
