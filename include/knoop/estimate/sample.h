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

#include <knoop/core.h>
#include <knoop/estimate/sample_estimator.h>
#include <knoop/io/hdf5.h>
#include <knoop/io/parse.h>
#include <knoop/sample/proband.h>

#include <Eigen/Dense>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

namespace knoop {

template <ProbandType P>
class SampleEstimatorStrategy {
 public:
  SampleEstimatorStrategy(
      const std::string& estimator_name,
      const std::string& sample_name,
      std::filesystem::path sample_dir,
      std::vector<std::string> row_labels,
      std::vector<std::string> col_labels,
      std::size_t n_rows,
      std::size_t n_cols = 1)
      : name_(estimator_name),
        name_h5_(std::format("{}/{}", sample_name, estimator_name)),
        sample_dir_(std::move(sample_dir)),
        row_labels_(std::move(row_labels)),
        col_labels_(std::move(col_labels)),
        n_rows_(n_rows),
        n_cols_(n_cols),
        data_(n_rows_, n_cols_) {
    HDF5Writer::create(data_, name_h5_, row_labels_, col_labels_);
  }

  virtual ~SampleEstimatorStrategy() = default;
  virtual void compute() = 0;

  std::string name() const { return name_; }
  std::vector<std::string> rowLabels() const { return row_labels_; }
  std::vector<std::string> colLabels() const { return col_labels_; }
  std::size_t numRows() const { return n_rows_; }
  std::size_t numCols() const { return n_cols_; }

  void operator()(std::size_t gen, std::size_t rep) {
    compute();
    HDF5Writer::write(data_, name_h5_, rep, gen);
  }

 protected:
  std::string name_;
  std::string name_h5_;
  std::filesystem::path sample_dir_;
  std::vector<std::string> row_labels_;
  std::vector<std::string> col_labels_;
  std::size_t n_rows_;
  std::size_t n_cols_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_;
};

template <ProbandType P>
class SampleEstimatorMeanStrategy : public SampleEstimatorStrategy<P> {
 public:
  explicit SampleEstimatorMeanStrategy(
      const Params& params,
      const std::string& sample_name,
      const std::filesystem::path& sample_dir)
      : SampleEstimatorStrategy<P>(
            "sample_mean",
            sample_name,
            sample_dir,
            {},
            params.pheno.names,
            params.pheno.n_pheno) {}

  void compute() override {
    Eigen::MatrixXd phenotypes =
        parsePhenoFile(this->sample_dir_ / "data.pheno");
    this->data_ = phenotypes.colwise().mean();
  }
};

template <ProbandType P>
class SampleEstimatorVarStrategy : public SampleEstimatorStrategy<P> {
 public:
  explicit SampleEstimatorVarStrategy(
      const Params& params,
      const std::string& sample_name,
      const std::filesystem::path& sample_dir)
      : SampleEstimatorStrategy<P>(
            "sample_var",
            sample_name,
            sample_dir,
            {},
            params.pheno.names,
            params.pheno.n_pheno) {}

  void compute() override {
    Eigen::MatrixXd phenotypes =
        parsePhenoFile(this->sample_dir_ / "data.pheno");
    this->data_ = (phenotypes.rowwise() - phenotypes.colwise().mean())
                      .array()
                      .square()
                      .colwise()
                      .sum() /
                  (phenotypes.rows() - 1);
  }
};

template <ProbandType P>
class SampleEstimatorCovStrategy : public SampleEstimatorStrategy<P> {
 public:
  explicit SampleEstimatorCovStrategy(
      const Params& params,
      const std::string& sample_name,
      const std::filesystem::path& sample_dir)
      : SampleEstimatorStrategy<P>(
            "sample_cov",
            sample_name,
            sample_dir,
            params.pheno.names,
            params.pheno.names,
            params.pheno.n_pheno,
            params.pheno.n_pheno),
        n_ind_(params.global.n_ind) {}

  void compute() override {
    Eigen::MatrixXd phenotypes =
        parsePhenoFile(this->sample_dir_ / "data.pheno");
    this->data_ =
        (phenotypes.rowwise() - phenotypes.colwise().mean()).transpose() *
        (phenotypes.rowwise() - phenotypes.colwise().mean()) /
        (static_cast<double>(phenotypes.rows() - 1));
  }

 private:
  std::size_t n_ind_;
};

template <ProbandType P>
class SampleEstimatorMateCorStrategy : public SampleEstimatorStrategy<P> {
 public:
  explicit SampleEstimatorMateCorStrategy(
      const Params& params,
      const std::string& sample_name,
      const std::filesystem::path& sample_dir,
      std::size_t n_probands)
      : SampleEstimatorStrategy<P>(
            "sample_mate_cor",
            sample_name,
            sample_dir,
            utils::vectorSuffix(params.pheno.names, "_male"),
            utils::vectorSuffix(params.pheno.names, "_female"),
            params.pheno.n_pheno,
            params.pheno.n_pheno),
        n_probands_(n_probands),
        n_pheno_(params.pheno.n_pheno),
        std_male_((n_probands_ * kProbandSize<P>) / 2, n_pheno_),
        std_female_((n_probands_ * kProbandSize<P>) / 2, n_pheno_) {
    if constexpr (P == ProbandType::Self) {
      throw std::runtime_error(
          "Unsupported proband type Proband::Self for estimator "
          "SampleMateCor");
    }
  }

  // compute centres and standard deviations across columns
  void compute() override {
    Eigen::MatrixXd phenotypes =
        parsePhenoFile(this->sample_dir_ / "data.pheno");
    const std::size_t n_pairs = n_probands_ * 3;
    const std::size_t outer = n_probands_ * kProbandSize<P>;

    using MateMat = Eigen::Map<
        const Eigen::MatrixXd,
        Eigen::Unaligned,
        Eigen::Stride<Eigen::Dynamic, 2>>;

    MateMat males(
        phenotypes.data(),
        n_pairs,
        n_pheno_,
        {static_cast<Eigen::Index>(outer), 2});

    MateMat females(
        phenotypes.data() + 1,
        n_pairs,
        n_pheno_,
        {static_cast<Eigen::Index>(outer), 2});

    utils::standardise(males, std_male_, false);
    utils::standardise(females, std_female_, false);

    this->data_ = (1.0 / (n_pairs - 1)) * std_male_.transpose() * std_female_;
  }

 private:
  std::size_t n_probands_;
  std::size_t n_pheno_;
  Eigen::MatrixXd std_male_;
  Eigen::MatrixXd std_female_;
};

template <ProbandType P>
inline SampleEstimator<P> sampleMean() {
  return SampleEstimator<P>{
      .name = "sample-mean",
      .fn = [](const Params& params,
               std::size_t /*n_probands*/,
               const std::filesystem::path& sample_dir) {
        return std::make_unique<SampleEstimatorMeanStrategy<P>>(
            params, sample_dir.filename().string(), sample_dir);
      }};
}

template <ProbandType P>
inline SampleEstimator<P> sampleVar() {
  return SampleEstimator<P>{
      .name = "sample-var",
      .fn = [](const Params& params,
               std::size_t /*n_probands*/,
               const std::filesystem::path& sample_dir) {
        return std::make_unique<SampleEstimatorVarStrategy<P>>(
            params, sample_dir.filename().string(), sample_dir);
      }};
}

template <ProbandType P>
inline SampleEstimator<P> sampleCov() {
  return SampleEstimator<P>{
      .name = "sample-cov",
      .fn = [](const Params& params,
               std::size_t /*n_probands*/,
               const std::filesystem::path& sample_dir) {
        return std::make_unique<SampleEstimatorCovStrategy<P>>(
            params, sample_dir.filename().string(), sample_dir);
      }};
}

template <ProbandType P>
inline SampleEstimator<P> sampleMateCor() {
  return SampleEstimator<P>{
      .name = "sample-mate-cor",
      .fn = [](const Params& params,
               std::size_t n_probands,
               const std::filesystem::path& sample_dir) {
        return std::make_unique<SampleEstimatorMateCorStrategy<P>>(
            params, sample_dir.filename().string(), sample_dir, n_probands);
      }};
}

}  // namespace knoop
