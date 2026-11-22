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
class SampleEstimatorGWASStrategy : public SampleEstimatorStrategy<P> {
 public:
  SampleEstimatorGWASStrategy(
      const Params& params,
      std::string sample_name,
      std::filesystem::path sample_dir,
      std::vector<std::string> pheno_names,
      std::string name = "gwas",
      std::size_t n_pcs = 0,
      double pval_threshold = 5e-8)
      : SampleEstimatorStrategy<P>(
            std::move(name),
            std::move(sample_name),
            sample_dir,
            {},
            {"pheno", "locus", "beta_est", "stderr", "pval", "code"},
            6,
            params.pheno.size() * params.geno.numLoci()),
        plink2_(std::move(process::resolveExecutable("plink2"))),
        n_pheno_(params.pheno.size()),
        n_loc_(params.geno.numLoci()),
        pval_threshold_(pval_threshold),
        n_pcs_(n_pcs),
        pheno_names_(std::move(pheno_names)),
        random_seed_(params.randomSeed()),
        effect_estimates_(params.geno.numLoci()),
        std_errors_(params.geno.numLoci()),
        p_values_(params.geno.numLoci()),
        effect_matrix_(
            Eigen::MatrixXd::Zero(params.geno.numLoci(), params.pheno.size())) {
    for (std::size_t p = 0; p < n_pheno_; ++p) {
      const auto& locs = params.pheno.causalLoci(p);
      const auto& effects = params.pheno.effects(p);
      for (std::size_t i = 0; i < locs.size(); ++i)
        effect_matrix_(locs[i], p) = effects(i);
    }
  }

  void compute() override {
    runGWAS();

    for (std::size_t p = 0; p < n_pheno_; ++p) {
      parseGWAS(p);
      auto rows = Eigen::seq(p * n_loc_, (p + 1) * n_loc_ - 1);
      this->data_(0, rows) =
          Eigen::VectorXd::Constant(n_loc_, static_cast<double>(p));
      this->data_(1, rows) = Eigen::VectorXd::LinSpaced(n_loc_, 0, n_loc_ - 1);
      this->data_(2, rows) = effect_estimates_;
      this->data_(3, rows) = std_errors_;
      this->data_(4, rows) = p_values_;

      for (std::size_t loc = 0; loc < n_loc_; ++loc) {
        if (p_values_(loc) >= pval_threshold_) {
          this->data_(5, (p * n_loc_) + loc) = 0.0;
        } else {
          this->data_(5, (p * n_loc_) + loc) =
              (effect_matrix_(loc, p) != 0.0) ? 1.0 : -1.0;
        }
      }
    }
  }

 private:
  bp::filesystem::path plink2_;
  std::size_t n_pheno_;
  std::size_t n_loc_;
  double pval_threshold_;
  std::size_t n_pcs_;
  std::vector<std::string> pheno_names_;
  std::size_t n_threads_;
  std::uint64_t random_seed_;

  Eigen::VectorXd effect_estimates_;
  Eigen::VectorXd std_errors_;
  Eigen::VectorXd p_values_;
  Eigen::MatrixXd effect_matrix_;

  Table<
      Column<"ID", std::string>,
      Column<"TEST", std::string>,
      Column<"A1", std::string>,
      Column<"BETA", double>,
      Column<"SE", double>,
      Column<"P", double>>
      table_;

  void runGWAS() {
    if (n_pcs_ > 0) {
      /**
       * if the number of requested principal components is non-zero, then run
       *    plink2 --bfile data --pca approx <n_pcs> --seed <seed> --threads
       *
       * TOOD: incorporate the --pheno-name flag to select the phenotypes:
       *
       *   plink2 --bfile data \
       *          --pca approx <n_pcs_> \
       *          --seed <random_seed_> \
       *          --threads <omp_get_num_threads()> \
       *          --pheno-names <pheno_names_>
       *
       * To use omp_get_num_threads() requires that knoop release / sleeps
       * OMP threads as soon as it exits its own replicate loop, meaning threads
       * can be reallocated to the spawned subprocess.
       */
      process::runProcess(
          plink2_,
          {"--bfile",
           (this->sample_dir_ / "data").string(),
           "--pca",
           "approx",
           std::format("{}", n_pcs_),
           "--seed",
           std::format("{}", random_seed_),
           "--threads",
           "1",
           "--out",
           (this->sample_dir_ / (this->name_ + "_pca")).string()});

      process::runProcess(
          plink2_,
          {"--bfile",
           (this->sample_dir_ / "data").string(),
           "--pheno",
           (this->sample_dir_ / "data.pheno").string(),
           "--covar",
           (this->sample_dir_ / (this->name_ + "_pca.eigenvec")).string(),
           "--glm",
           "--variance-standardize",
           "--no-psam-pheno",
           "--threads",
           "1",
           "--out",
           (this->sample_dir_ / this->name_).string()});

    } else {
      process::runProcess(
          plink2_,
          {"--bfile",
           (this->sample_dir_ / "data").string(),
           "--pheno",
           (this->sample_dir_ / "data.pheno").string(),
           "--glm",
           "allow-no-covars",
           "--variance-standardize",
           "--no-psam-pheno",
           "--threads",
           "1",
           "--out",
           (this->sample_dir_ / this->name_).string()});
    }
  }

  void parseGWAS(std::size_t pheno_index);
};

template <ProbandType P>
inline void SampleEstimatorGWASStrategy<P>::parseGWAS(std::size_t pheno_index) {
  table_.readFile(
      this->sample_dir_ /
          std::format(
              "{}.{}.glm.linear", this->name_, pheno_names_[pheno_index]),
      '\t');

  for (std::size_t loc = 0; loc < n_loc_; ++loc) {
    auto [id, test, a1, effect_estimate, stderr, pval] = table_.row(loc);
    effect_estimates_(loc) = effect_estimate;
    std_errors_(loc) = stderr;
    p_values_(loc) = pval;
  }
}

inline SampleEstimator sampleGWAS(
    const std::string& name = "gwas",
    std::size_t n_pcs = 0,
    double pval_threshold = 5e-8) {
  return SampleEstimator([name, n_pcs, pval_threshold]<ProbandType P>() {
    return SampleEstimatorFor<P>{
        .name = name,
        .fn = [name = name, n_pcs, pval_threshold](
                  const Params& params,
                  std::size_t /*n_probands*/,
                  const std::filesystem::path& sample_dir) {
          return std::make_unique<SampleEstimatorGWASStrategy<P>>(
              params,
              sample_dir.filename().string(),
              sample_dir,
              params.pheno.names(),
              name,
              n_pcs,
              pval_threshold);
        }};
  });
}

}  // namespace knoop
