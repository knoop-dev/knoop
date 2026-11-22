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

#include <knoop/core/error.h>

#include <Eigen/Dense>
#include <cstddef>
#include <set>
#include <vector>

namespace knoop {

inline bool checkIndicesUnique(const std::vector<std::size_t>& indices) {
  std::set<std::size_t> indices_set;

  for (const std::size_t& index : indices) {
    if (!indices_set.contains(index))
      indices_set.insert(index);
    else
      return false;
  }

  return true;
}

inline bool checkIndicesInRange(
    const std::vector<std::size_t>& indices,
    std::size_t lower,
    std::size_t upper,
    bool lower_inclusive = true,
    bool upper_inclusive = true) {
  bool result = true;
  for (const std::size_t& index : indices) {
    bool lower_inclusion = lower_inclusive ? index >= lower : index > lower;
    bool upper_inclusion = upper_inclusive ? index <= upper : index < upper;
    result &= lower_inclusion && upper_inclusion;
  }
  return result;
}

inline bool checkVectorInRange(
    const Eigen::VectorXd& vector,
    double lower = std::numeric_limits<double>::min(),
    double upper = std::numeric_limits<double>::max(),
    bool lower_inclusive = true,
    bool upper_inclusive = true) {
  bool result = true;
  for (const double& value : vector) {
    bool lower_inclusion = lower_inclusive ? value >= lower : value > lower;
    bool upper_inclusion = upper_inclusive ? value <= upper : value < upper;
    result &= lower_inclusion && upper_inclusion;
  }
  return result;
}

inline bool checkMatrixSymmetric(const Eigen::MatrixXd& mat) {
  return mat.isApprox(mat.transpose());
}

inline bool checkMatrixPSD(const Eigen::MatrixXd& mat) {
  if (!checkMatrixSymmetric(mat)) return false;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> evd(mat);

  if (evd.info() != Eigen::Success)
    throw KnoopNumericalError("Eigendecomposition of matrix failed");

  return checkVectorInRange(evd.eigenvalues(), 0);
}

inline bool checkMatrixPD(const Eigen::MatrixXd& mat) {
  if (!checkMatrixSymmetric(mat)) return false;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> evd(mat);

  if (evd.info() != Eigen::Success)
    throw KnoopNumericalError("Eigendecomposition of matrix failed");

  return checkVectorInRange(
      evd.eigenvalues(), 0, std::numeric_limits<double>::max(), false);
}

}  // namespace knoop
