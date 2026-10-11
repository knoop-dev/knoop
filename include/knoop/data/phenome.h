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
#include <knoop/data/component.h>

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace knoop {

// Buffer structure to store phenotype data
class PhenotypeBuffer {
 public:
  explicit PhenotypeBuffer(const Params& params)
      : n_ind_(params.global.n_ind),
        n_sex_(n_ind_ / 2),
        n_pheno_(params.pheno.n_pheno),
        data_(4 * n_ind_ * n_pheno_),
        comp_mean_(4, n_pheno_),
        comp_var_(4, n_pheno_) {}

  Eigen::Map<const Eigen::MatrixXd> operator()() const {
    const double* pos = data_.data();
    return Eigen::Map<const Eigen::MatrixXd>(pos, n_ind_, n_pheno_ * 4);
  }

  Eigen::Map<Eigen::MatrixXd> operator()() {
    double* pos = data_.data();
    return Eigen::Map<Eigen::MatrixXd>(pos, n_ind_, n_pheno_ * 4);
  }

  // Retrieve const matrix of values for specified component
  Eigen::Map<const Eigen::MatrixXd> operator()(
      Component type = Component::Total) const {
    const double* pos = &data_[n_ind_ * (n_pheno_ * static_cast<int>(type))];
    return Eigen::Map<const Eigen::MatrixXd>(pos, n_ind_, n_pheno_);
  }

  // Retrieve matrix of values for specified component
  Eigen::Map<Eigen::MatrixXd> operator()(Component type = Component::Total) {
    double* pos = &data_[n_ind_ * (n_pheno_ * static_cast<int>(type))];
    return Eigen::Map<Eigen::MatrixXd>(pos, n_ind_, n_pheno_);
  }

  // Retrieve const vector of values for specified phenotype component
  Eigen::Map<const Eigen::VectorXd> operator()(
      std::size_t id, Component type = Component::Total) const {
    const double* pos =
        &data_[n_ind_ * (n_pheno_ * static_cast<int>(type) + id)];
    return Eigen::Map<const Eigen::VectorXd>(pos, n_ind_);
  }

  // Retrieve vector of values for specified phenotype component
  Eigen::Map<Eigen::VectorXd> operator()(
      std::size_t id, Component type = Component::Total) {
    double* pos = &data_[n_ind_ * (n_pheno_ * static_cast<int>(type) + id)];
    return Eigen::Map<Eigen::VectorXd>(pos, n_ind_);
  }

  auto male(Component type = Component::Total) {
    Eigen::OuterStride<> stride(n_ind_);
    double* pos = &data_[n_ind_ * (n_pheno_ * static_cast<int>(type))];
    return Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(
        pos, n_sex_, n_pheno_, stride);
  }

  auto female(Component type = Component::Total) {
    Eigen::OuterStride<> stride(n_ind_);
    double* pos =
        &data_[(n_ind_ * (n_pheno_ * static_cast<int>(type))) + (n_ind_ / 2)];
    return Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(
        pos, n_sex_, n_pheno_, stride);
  }

  Eigen::Map<Eigen::VectorXd> male(
      std::size_t id, Component type = Component::Total) {
    double* pos = &data_[n_ind_ * (n_pheno_ * static_cast<int>(type) + id)];
    return Eigen::Map<Eigen::VectorXd>(pos, n_sex_);
  }

  Eigen::Map<Eigen::VectorXd> female(
      std::size_t id, Component type = Component::Total) {
    double* pos =
        &data_[(n_ind_ * (n_pheno_ * static_cast<int>(type) + id)) + n_sex_];
    return Eigen::Map<Eigen::VectorXd>(pos, n_sex_);
  }

  auto male(Component type = Component::Total) const {
    Eigen::OuterStride<> stride(n_ind_);
    const double* pos = &data_[n_ind_ * (n_pheno_ * static_cast<int>(type))];
    return Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>(
        pos, n_sex_, n_pheno_, stride);
  }

  auto female(Component type = Component::Total) const {
    Eigen::OuterStride<> stride(n_ind_);
    const double* pos =
        &data_[(n_ind_ * (n_pheno_ * static_cast<int>(type))) + (n_ind_ / 2)];
    return Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>(
        pos, n_sex_, n_pheno_, stride);
  }

  Eigen::Map<const Eigen::VectorXd> male(
      std::size_t id, Component type = Component::Total) const {
    const double* pos =
        &data_[n_ind_ * (n_pheno_ * static_cast<int>(type) + id)];
    return Eigen::Map<const Eigen::VectorXd>(pos, n_sex_);
  }

  Eigen::Map<const Eigen::VectorXd> female(
      std::size_t id, Component type = Component::Total) const {
    const double* pos =
        &data_[(n_ind_ * (n_pheno_ * static_cast<int>(type) + id)) + n_sex_];
    return Eigen::Map<const Eigen::VectorXd>(pos, n_sex_);
  }

  // means and variances of phenotype components
  void computeStats();

  double componentMean(std::size_t pheno_id, Component type) const {
    return comp_mean_(static_cast<int>(type), pheno_id);
  }

  double componentVar(std::size_t pheno_id, Component type) const {
    return comp_var_(static_cast<int>(type), pheno_id);
  }

  Eigen::Ref<const Eigen::VectorXd> componentMean(Component type) const {
    return comp_mean_.row(static_cast<int>(type));
  }

  Eigen::Ref<const Eigen::VectorXd> componentVar(Component type) const {
    return comp_var_.row(static_cast<int>(type));
  }

 private:
  const std::size_t n_ind_;
  const std::size_t n_sex_;
  const std::size_t n_pheno_;
  std::vector<double> data_;

  Eigen::MatrixXd comp_mean_;
  Eigen::MatrixXd comp_var_;
};

inline void PhenotypeBuffer::computeStats() {
  for (Component type :
       {Component::Genetic,
        Component::Environmental,
        Component::Vertical,
        Component::Total}) {
    Eigen::Index col_type = static_cast<int>(type);
    comp_mean_.row(col_type) = (*this)(type).colwise().mean();
    comp_var_.row(col_type) =
        ((*this)(type).rowwise() - comp_mean_.row(col_type))
            .array()
            .square()
            .colwise()
            .sum() /
        static_cast<double>(n_ind_);
  }
}

}  // namespace knoop
