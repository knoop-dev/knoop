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
      std::string name = "gwas",
      std::size_t n_pcs = 0,
      double pval_threshold = 5e-8)
      : SampleEstimatorStrategy<P>(
            std::move(name),
            std::move(sample_name),
            sample_dir,
            params.pheno.names,
            {"l2_effect", "fpr", "tpr", "pgs_r2", "pgs_rmse"},
            params.pheno.n_pheno,
            5),
        plink2_(std::move(process::resolveExecutable("plink2"))),
        n_pheno_(params.pheno.n_pheno),
        n_loc_(params.geno.n_loc),
        pval_threshold_(pval_threshold),
        n_pcs_(n_pcs),
        pheno_names_(params.pheno.names),
        random_seed_(params.global.random_seed),
        beta_true_(n_pheno_, Eigen::VectorXd::Zero(n_loc_)),
        causal_mask_(n_pheno_, Eigen::VectorXd::Zero(n_loc_)) {
    for (std::size_t p = 0; p < n_pheno_; ++p)
      for (std::size_t i = 0; i < params.pheno.pheno_loc[p].size(); ++i) {
        std::size_t loc = params.pheno.pheno_loc[p][i];
        beta_true_[p](loc) = params.pheno.pheno_effects[p](i);
        causal_mask_[p](loc) = 1.0;
      }
  }

  void compute() override {
    runGWAS();

    for (std::size_t p = 0; p < n_pheno_; ++p) {
      auto [betas, pvals] = parseGWAS(p);

      auto sig = (pvals.array() < pval_threshold_).template cast<double>();
      auto& mask = causal_mask_[p];
      double n_causal = mask.sum();
      double n_null = n_loc_ - n_causal;

      double l2 = (betas - beta_true_[p]).norm();
      double tpr = n_causal > 0 ? (sig * mask.array()).sum() / n_causal : 0.0;
      double fpr =
          n_null > 0 ? (sig * (1.0 - mask.array())).sum() / n_null : 0.0;
      auto [r2, rmse] = pgsMetrics(p);

      this->data_.row(p) << l2, fpr, tpr, r2, rmse;
    }
  }

 private:
  bp::filesystem::path plink2_;
  std::size_t n_pheno_;
  std::size_t n_loc_;
  double pval_threshold_;
  std::size_t n_pcs_;
  std::vector<std::string> pheno_names_;
  std::uint64_t random_seed_;

  Table<
      Column<"ID", std::string>,
      Column<"TEST", std::string>,
      Column<"A1", std::string>,
      Column<"BETA", double>,
      Column<"P", double>>
      table_;

  std::vector<Eigen::VectorXd> beta_true_;
  std::vector<Eigen::VectorXd> causal_mask_;

  void runGWAS() {
    if (n_pcs_ > 0) {
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

  std::pair<Eigen::VectorXd, Eigen::VectorXd> parseGWAS(std::size_t p) {
    Eigen::VectorXd betas = Eigen::VectorXd::Zero(n_loc_);
    Eigen::VectorXd pvals = Eigen::VectorXd::Ones(n_loc_);

    table_.readFile(
        this->sample_dir_ /
            std::format("{}.{}.glm.linear", this->name_, pheno_names_[p]),
        '\t');

    auto hits = table_.filter([&](const auto& row) {
      auto [id, test, a1, beta, pval] = row;
      return test == "ADD";
    });

    for (std::size_t r = 0; r < hits.numRows(); ++r) {
      auto [id, test, a1, beta, pval] = hits.row(r);
      std::size_t loc = parse<std::size_t>(id.substr(3));
      betas(loc) = beta;
      pvals(loc) = pval;
    }

    return {betas, pvals};
  }

  std::pair<double, double> pgsMetrics(std::size_t p) {
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    auto score_file =
        this->sample_dir_ / std::format("{}_score_{}.txt", this->name_, p);
    auto pfix = this->sample_dir_ / std::format("{}_pgs_{}", this->name_, p);

    auto hits = table_.filter([&](const auto& row) {
      auto [id, test, a1, beta, pval] = row;
      return test == "ADD" && pval < pval_threshold_;
    });

    if (hits.empty()) {
      Log::warning(
          "{}: no genome-wide-significant hits for phenotype '{}' "
          "(pval < {}) — pgs_r2/pgs_rmse will be 0.0/NaN",
          this->name_,
          pheno_names_[p],
          pval_threshold_);
      return {0.0, NaN};
    }

    writeTable<"ID", "A1", "BETA">(hits, score_file, '\t', '\n');

    // based on GWAS hits, construct a polygenic score
    process::runProcess(
        plink2_,
        {"--bfile",
         (this->sample_dir_ / "data").string(),
         "--score",
         score_file.string(),
         "1",
         "2",
         "3",
         "header",
         "--threads",
         "1",
         "--out",
         pfix.string()});

    auto sscore_path = pfix.string() + ".sscore";
    auto gen_path = this->sample_dir_ / "data.genetic.pheno";

    if (!std::filesystem::exists(sscore_path)) {
      Log::error(
          "{}: expected plink2 --score output not found: {}",
          this->name_,
          sscore_path);
      return {NaN, NaN};
    }

    if (!std::filesystem::exists(gen_path)) {
      Log::error(
          "{}: expected genetic-value file not found: {}",
          this->name_,
          gen_path.string());
      return {NaN, NaN};
    }

    // read in computed pgs .sscore file
    Table<Column<"IID", std::string>, Column<"SCORE1_AVG", double>> pgs_table;
    pgs_table.readFile(sscore_path, '\t');
    if (pgs_table.empty()) {
      Log::error(
          "{}: {} exists but contains no scored individuals for "
          "phenotype '{}'",
          this->name_,
          sscore_path,
          pheno_names_[p]);
      return {0.0, NaN};
    }

    // read in genetic component of phenotype written out by sampling.h
    TableXd<Column<"FID", std::string>, Column<"IID", std::string>> gen_table;
    gen_table.readFile(gen_path, '\t');
    if (gen_table.known().empty() ||
        gen_table.matrix().cols() <= static_cast<Eigen::Index>(p)) {
      Log::error(
          "{}: {} has no data for phenotype '{}' (index {})",
          this->name_,
          gen_path.string(),
          pheno_names_[p],
          p);
      return {NaN, NaN};
    }

    std::vector<double> pgs_vals = getColumn<"SCORE1_AVG">(pgs_table);
    Eigen::VectorXd gen = gen_table.matrix().col(p);
    if (pgs_vals.empty() ||
        pgs_vals.size() != static_cast<std::size_t>(gen.size())) {
      Log::error(
          "{}: individual count mismatch between {} ({} rows) and {} "
          "({} rows) for phenotype '{}'",
          this->name_,
          sscore_path,
          pgs_vals.size(),
          gen_path.string(),
          gen.size(),
          pheno_names_[p]);
      return {NaN, NaN};
    }

    Eigen::VectorXd pgs =
        Eigen::Map<Eigen::VectorXd>(pgs_vals.data(), pgs_vals.size());

    auto pc = pgs.array() - pgs.mean();
    auto gc = gen.array() - gen.mean();
    double num = (pc * gc).sum();
    double r2 = (num * num) / (pc.square().sum() * gc.square().sum());
    double rmse = std::sqrt((pgs - gen).squaredNorm() / pgs.size());
    return {r2, rmse};
  }
};

template <ProbandType P>
inline SampleEstimator<P> sampleGWAS(
    std::string name = "gwas",
    std::size_t n_pcs = 0,
    double pval_threshold = 5e-8) {
  return SampleEstimator<P>{
      .name = name,
      .fn = [name = std::move(name), n_pcs, pval_threshold](
                const Params& params,
                std::size_t /*n_probands*/,
                const std::filesystem::path& sample_dir) {
        return std::make_unique<SampleEstimatorGWASStrategy<P>>(
            params,
            sample_dir.filename().string(),
            sample_dir,
            name,
            n_pcs,
            pval_threshold);
      }};
}

}  // namespace knoop
