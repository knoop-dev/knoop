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
#include <knoop/core/random.h>
#include <sys/wait.h>

#include <Eigen/Dense>
#include <boost/process.hpp>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace knoop::utils {

template <typename T>
inline std::string vectorToString(const T& vector) {
  return std::accumulate(
      vector.begin() + 1,
      vector.end(),
      std::format("{:g}", vector[0]),
      [](const std::string& a, double b) {
        return a + "," + std::format("{:g}", b);
      });
}

inline std::vector<std::string> splitString(
    const std::string& s, const char& delim = ',', bool remove_empty = true) {
  std::vector<std::string> split;
  boost::split(split, s, boost::is_any_of(std::string(1, delim)));
  if (remove_empty)
    std::erase_if(split, [](const std::string& s) { return s.empty(); });
  return split;
}

inline std::vector<std::string> vectorPrefix(
    const std::vector<std::string>& labels,
    const std::optional<std::string>& prefix = std::nullopt) {
  std::vector<std::string> result(labels.size());
  for (std::size_t i = 0; i < labels.size(); ++i)
    result[i] = prefix.value_or("") + labels[i];
  return result;
}

inline std::vector<std::string> vectorSuffix(
    const std::vector<std::string>& labels,
    const std::optional<std::string>& suffix = std::nullopt) {
  std::vector<std::string> result(labels.size());
  for (std::size_t i = 0; i < labels.size(); ++i)
    result[i] = labels[i] + suffix.value_or("");
  return result;
}

inline Eigen::MatrixXd randomOrthogonal(std::size_t dim) {
  Eigen::MatrixXd random(dim, dim);
  random::Normal::sample(
      random::Xoshiro256pp::getInstance(), random.data(), dim * dim);

  Eigen::HouseholderQR<Eigen::MatrixXd> qr(random);
  return qr.householderQ();
}

inline Eigen::MatrixXd matrixFromSingularValues(
    const std::vector<double>& values, bool symmetric = false) {
  std::size_t n_pheno = values.size();
  Eigen::MatrixXd u_mat;
  Eigen::MatrixXd v_mat;

  Eigen::VectorXd singular_values(n_pheno);
  for (std::size_t i = 0; i < n_pheno; ++i) singular_values(i) = values[i];

  Eigen::MatrixXd s_mat = singular_values.asDiagonal();

  u_mat = randomOrthogonal(n_pheno);
  if (!symmetric) v_mat = randomOrthogonal(n_pheno);

  return (symmetric) ? u_mat * s_mat * u_mat.transpose()
                     : u_mat * s_mat * v_mat.transpose();
}

template <typename EigenMatrixType>
inline void standardise(
    const Eigen::MatrixXd& mat,
    EigenMatrixType& mat_into,
    bool population = true,
    bool centre = true,
    bool scale = true) {
  if (!centre && !scale) {
    mat_into = mat;
    return;
  }
  Eigen::RowVectorXd mean = centre ? Eigen::RowVectorXd(mat.colwise().mean())
                                   : Eigen::RowVectorXd::Zero(mat.cols());
  mat_into = mat.rowwise() - mean;
  if (!scale) return;

  Eigen::RowVectorXd std =
      ((mat.rowwise() - mean).array().square().colwise().sum() /
       (population ? mat.rows() : mat.rows() - 1))
          .sqrt()
          .max(1e-10);
  mat_into = ((mat.rowwise() - mean).array().rowwise() / std.array());
}

inline std::vector<std::size_t> order(const Eigen::VectorXd& v) {
  std::vector<std::size_t> idx(v.size());
  // NOLINTBEGIN(modernize-use-ranges)
  std::iota(idx.begin(), idx.end(), 0);
  // NOLINTEND(modernize-use-ranges)
  std::ranges::stable_sort(
      idx, [&v](std::size_t i0, std::size_t i1) { return v(i0) < v(i1); });
  return idx;
}

}  // namespace knoop::utils
