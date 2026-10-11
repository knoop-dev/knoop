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
#include <knoop/io/table.h>
#include <knoop/sample/proband.h>

#include <Eigen/Dense>
#include <boost/process.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>

namespace knoop {

namespace bp = boost::process;

template <ProbandType P>
class SampleEstimatorHasemanElstonStrategy : public SampleEstimatorStrategy<P> {
 public:
  SampleEstimatorHasemanElstonStrategy(
      const Params& params,
      std::string sample_name,
      std::filesystem::path sample_dir,
      std::string name = "haseman-elston")
      : SampleEstimatorStrategy<P>(
            std::move(name),
            std::move(sample_name),
            sample_dir,
            params.pheno.names,
            std::vector<std::string>{"V(G)/Vp"},
            params.pheno.n_pheno,
            1),
        gcta64_(std::move(process::resolveExecutable("gcta64"))),
        n_pheno_(params.pheno.n_pheno) {}

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
      auto pfix = this->sample_dir_ / std::format("he_{}", p);
      process::runProcess(
          gcta64_,
          {"--grm",
           (this->sample_dir_ / "grm"),
           "--pheno",
           (this->sample_dir_ / "data.pheno"),
           "--mpheno",
           std::format("{}", p + 1),
           "--HEreg",
           "--out",
           pfix.string(),
           "--thread-num",
           "1"});

      this->data_(p, 0) = parseHEreg(pfix.string() + ".HEreg", p);
    }
  }

 private:
  bp::filesystem::path gcta64_;
  std::size_t n_pheno_;

  double parseHEreg(const std::string& path, std::size_t p) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    if (!std::filesystem::exists(path)) {
      Log::error(
          "{}: gcta64 --HEreg exited successfully but {} was never "
          "written, for phenotype index {}",
          this->name_,
          path,
          p);
      return kNaN;
    }

    /**
     * .HEreg stacks two method blocks (HE-CP, HE-SD), each its own label
     * line + header + rows, separated by a blank line — we only want HE-CP
     * (the cross-product method, analogous to GREML heritability).
     */
    std::ifstream file(path);
    std::string contents(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    std::size_t block_end = contents.find("\n\n");
    std::string he_cp_block = contents.substr(0, block_end);

    std::size_t header_start = he_cp_block.find('\n') + 1;
    std::string he_cp_table = he_cp_block.substr(header_start);

    std::filesystem::path tmp_path = path + ".he-cp";
    std::ofstream tmp(tmp_path);
    tmp << he_cp_table;
    tmp.close();

    Table<Column<"Coefficient", std::string>, Column<"Estimate", double>> hereg;
    hereg.readFile(tmp_path, ' ');
    std::filesystem::remove(tmp_path);

    for (std::size_t r = 0; r < hereg.numRows(); ++r) {
      auto [source, variance] = hereg.row(r);
      if (source == "V(G)/Vp") return variance;
    }

    Log::error(
        "{}: {} parsed but 'V(G)/Vp' was not found in HE-CP for "
        "phenotype index {}",
        this->name_,
        path,
        p);
    return kNaN;
  }
};

template <ProbandType P>
inline SampleEstimator<P> sampleHasemanElston(
    std::string name = "haseman-elston") {
  return SampleEstimator<P>{
      .name = name,
      .fn = [name = std::move(name)](
                const Params& params,
                std::size_t /*n_probands*/,
                const std::filesystem::path& sample_dir) {
        return std::make_unique<SampleEstimatorHasemanElstonStrategy<P>>(
            params, sample_dir.filename().string(), sample_dir, name);
      }};
}

}  // namespace knoop
