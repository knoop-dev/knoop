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

#include <format>
#define SOL_ALL_SAFETIES_ON 1
#include <Eigen/Dense>
#include <sol/sol.hpp>

template <>
struct sol::is_container<Eigen::MatrixXd> : std::false_type {};

template <>
struct sol::is_automagical<Eigen::MatrixXd> : std::false_type {};

/**
 * This file should register all operations and numerical types that the user
 * may find convenient.
 *
 * They should all WRAP Eigen's types. So:
 *
 * local vec = Vector{1, 2, 3}
 *
 * local phenotypes = state:phenotypes{gen = -1}
 *
 * phenotypes.T * phenotypes -- phenotype cor matrix in parental generation
 *
 * local genotypes = state:genotypes{gen = -1} -- this is a GenotypeMatrix
 *
 * genotypes.T * genotypes
 *
 * genotypes * effects -- effects must be a Vector type whose dimension matches
 *
 * Let's take two vectors,
 *
 * local a = Vector{...}
 * local b = Vector{...}
 *
 * We should be able to do:
 *
 * a.dot(b)
 *
 * a.T * b
 *
 * a + b
 *
 * a:norm()
 *
 * a / c -- here c is a scalar
 *
 * log(a) -- a.array().log()
 *
 * mean(a)
 *
 * std(a)
 */

namespace knoop::lua {

inline void defineEigenBindings(sol::state& lua) {
  /**
   * Matrices are missing!
   */
  lua.new_usertype<Eigen::MatrixXd>(
      "matrix",
      sol::call_constructor,
      sol::factories([](const sol::table& tbl) {
        if (tbl.size() == 0) return Eigen::MatrixXd(0, 0);

        /** check for whether the user is specifying R-style */
        if (tbl[1].is<Eigen::VectorXd>()) {
          auto vec = tbl.get<Eigen::VectorXd>(1);

          std::size_t n_rows = tbl["nrow"].valid()
                                   ? tbl.get<std::size_t>("nrow")
                                   : tbl.get<std::size_t>(3);

          std::size_t n_cols = tbl["ncol"].valid()
                                   ? tbl.get<std::size_t>("ncol")
                                   : tbl.get<std::size_t>(3);

          if (vec.size() != n_rows * n_cols)
            throw KnoopDimensionError(
                "Expected vector size {} (got: {})",
                n_rows * n_cols,
                vec.size());

          Eigen::MatrixXd mat =
              Eigen::Map<Eigen::MatrixXd>(vec.data(), n_rows, n_cols);

          return mat;
        }

        /** consistenty checks */
        std::size_t n_rows = tbl.size();
        std::size_t n_cols = tbl.get<sol::table>(1).size();

        /** CARE: Lua is 1-indexed! */
        Eigen::MatrixXd mat(n_rows, n_cols);
        for (std::size_t row = 0; row < n_rows; ++row) {
          if (tbl.get<sol::table>(row + 1).size() != n_cols)
            throw KnoopDimensionError(
                "Number of columns must be consistent across rows");

          for (std::size_t col = 0; col < n_cols; ++col) {
            mat(row, col) = tbl[row + 1][col + 1];
          }
        }

        return mat;
      }),
      sol::meta_function::equal_to,
      [](const Eigen::MatrixXd& matA, const Eigen::MatrixXd& matB) {
        return matA.isApprox(matB);
      },
      sol::meta_function::length,
      [](const Eigen::MatrixXd& mat) { return mat.rows(); },
      sol::meta_function::to_string,
      [](const Eigen::MatrixXd& mat) {
        std::stringstream ss;
        ss << mat;
        return ss.str();
      },
      sol::meta_function::concatenation,
      sol::overload(
          [](const Eigen::MatrixXd& mat, const std::string& str) {
            std::stringstream ss;
            ss << mat.transpose();
            return str + ss.str();
          },
          [](const std::string& str, const Eigen::MatrixXd& mat) {
            std::stringstream ss;
            ss << mat.transpose();
            return str + ss.str();
          }),
      sol::meta_function::index,
      sol::overload(
          [](const Eigen::MatrixXd& self, std::size_t row) {
            return Eigen::VectorXd(self.row(row - 1));
          },
          [](const Eigen::MatrixXd& self, std::size_t row, std::size_t col) {
            return self(row - 1, col - 1);
          }),
      sol::meta_function::multiplication,
      sol::overload(
          [](const Eigen::MatrixXd& matA, const Eigen::MatrixXd& matB) {
            if (matA.cols() != matB.rows())
              throw KnoopNumericalError(
                  "Incomformable matrix multiplication; arguments have "
                  "dimensions {}x{} and {}x{}",
                  matA.rows(),
                  matA.cols(),
                  matB.rows(),
                  matB.cols());
            return Eigen::MatrixXd(matA * matB);
          },
          [](const Eigen::MatrixXd& matA, double v) {
            return Eigen::MatrixXd(v * matA);
          },
          [](double v, const Eigen::MatrixXd& matA) {
            return Eigen::MatrixXd(v * matA);
          }),
      sol::meta_function::addition,
      [](const Eigen::MatrixXd& matA, const Eigen::MatrixXd& matB) {
        if (matA.rows() != matB.rows() || matA.cols() != matB.cols())
          throw KnoopNumericalError(
              "Incomformable matrix addition; arguments have dimensions "
              "{}x{} and {}x{}",
              matA.rows(),
              matA.cols(),
              matB.rows(),
              matB.cols());
        return Eigen::MatrixXd(matA + matB);
      },
      "row",
      [](const Eigen::MatrixXd& self, std::size_t row) {
        return Eigen::VectorXd(self.row(row - 1));
      },
      "col",
      [](const Eigen::MatrixXd& self, std::size_t col) {
        return Eigen::VectorXd(self.col(col - 1));
      },
      "at",
      [](const Eigen::MatrixXd& self, std::size_t row, std::size_t col) {
        return self(row - 1, col - 1);
      },
      "T",
      sol::property([](const Eigen::MatrixXd& mat) {
        return Eigen::MatrixXd(mat.transpose());
      }));

  // something – scope out Eigen docs for how to do this
  lua.set_function(
      "solve",
      sol::overload(
          [](const Eigen::MatrixXd& mat, const Eigen::VectorXd& vec) {
            return Eigen::VectorXd(mat.colPivHouseholderQr().solve(vec));
          },
          [](const Eigen::MatrixXd& mat) {
            std::size_t dim = mat.rows();

            if (mat.rows() != mat.cols())
              throw knoop::KnoopInvalidArgumentError(
                  "Can not compute inverse of non-square matrix");

            if (!mat.fullPivLu().isInvertible())
              throw KnoopNumericalError("Matrix is not invertible");

            return Eigen::MatrixXd(
                mat.partialPivLu().solve(Eigen::MatrixXd::Identity(dim, dim)));
          }));

  lua.set_function("inverse", [](const Eigen::MatrixXd& mat) {
    if (!mat.fullPivLu().isInvertible())
      throw KnoopNumericalError("Matrix is not invertible");
    return Eigen::MatrixXd(mat.inverse());
  });

  lua.set_function("chol", [](const Eigen::MatrixXd& mat) {
    if (!mat.isApprox(mat.transpose()))
      throw knoop::KnoopInvalidArgumentError(
          "Matrix must be symmetric to perform Cholesky decomposition");

    Eigen::LLT<Eigen::MatrixXd> llt(mat);

    if (llt.info() != Eigen::Success)
      throw KnoopNumericalError("Matrix is not positive definite");

    return Eigen::MatrixXd(llt.matrixL());
  });

  // eigendecomposition, return eigenvalues as vectors, eigenvectors as matrix
  lua.set_function("eigen", [](const Eigen::MatrixXd& mat) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> evd(mat);
    return std::make_tuple(evd.eigenvalues(), evd.eigenvectors());
  });

  // singular value decomposition
  lua.set_function("svd", [](const Eigen::MatrixXd& mat) {
    Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV>
        svd(mat);
    return std::make_tuple(svd.singularValues(), svd.matrixU(), svd.matrixV());
  });

  lua.set_function("qr", [](const Eigen::MatrixXd& mat) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(mat);
    Eigen::MatrixXd q =
        qr.householderQ() * Eigen::MatrixXd::Identity(mat.rows(), mat.cols());
    Eigen::MatrixXd r = qr.matrixQR().triangularView<Eigen::Upper>();

    return std::make_tuple(q, r);
  });

  /**
   * Desired ops, for matrices or vectors:
   * - matrix multiplication,
   * - scalar addition,
   * - length / dimension,
   * - logarithms,
   * - exponents,
   * - elementwise sqrt (sqrt)
   * - summation,
   * - mean,
   * - standard deviation,
   * - principal square (sqrtm),
   * - cholesky (chol),
   * - eigendecomposition (eigen),
   * - trace,
   * - determinant,
   * - norms,
   * - indexing / slicing,
   * - solve Ax = b (solve(Ax, b)), also used for inverse
   * - ... ?
   */
  lua.new_usertype<Eigen::VectorXd>(
      "vector",
      sol::call_constructor,
      sol::factories([](const sol::table& tbl) {
        Eigen::VectorXd vec(tbl.size());
        for (std::size_t el = 0; el < tbl.size(); ++el) vec(el) = tbl[el + 1];
        return vec;
      }),
      sol::meta_function::equal_to,
      [](const Eigen::VectorXd& vecA, const Eigen::VectorXd& vecB) {
        return vecA.isApprox(vecB);
      },
      sol::meta_function::length,
      [](const Eigen::VectorXd& vec) { return vec.size(); },
      sol::meta_function::to_string,
      [](const Eigen::VectorXd& vec) {
        std::stringstream ss;
        ss << vec;
        return ss.str();
      },
      sol::meta_function::concatenation,
      sol::overload(
          [](const Eigen::VectorXd& vec, const std::string& str) {
            std::stringstream ss;
            ss << vec.transpose();
            return str + ss.str();
          },
          [](const std::string& str, const Eigen::VectorXd& vec) {
            std::stringstream ss;
            ss << vec.transpose();
            return str + ss.str();
          }),
      sol::meta_function::index,
      [](const Eigen::VectorXd& self, std::size_t index) {
        return self(index - 1);
      },
      sol::meta_function::addition,
      [](const Eigen::VectorXd& vecA, const Eigen::VectorXd& vecB) {
        if (vecA.size() != vecB.size())
          throw KnoopInvalidArgumentError("Vectors must be of equal length");

        return Eigen::VectorXd(vecA + vecB);
      },
      "T",
      sol::property([](const Eigen::VectorXd& vec) {
        return Eigen::RowVectorXd(vec.transpose());
      }),
      "dot",
      [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a.dot(b);
      },
      sol::meta_function::multiplication,
      sol::overload(
          [](const Eigen::RowVectorXd& a, const Eigen::VectorXd& b) {
            return a.dot(b);
          },
          [](const Eigen::VectorXd& a, double b) {
            return Eigen::VectorXd(a * b);
          },
          [](double b, const Eigen::VectorXd& a) {
            return Eigen::VectorXd(b * a);
          },
          [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            return Eigen::VectorXd(a.cwiseProduct(b));
          }),
      sol::meta_function::division,
      [](const Eigen::VectorXd& a, double b) { return Eigen::VectorXd(a / b); },
      sol::meta_function::division,
      [](double a, const Eigen::VectorXd& vec) {
        return Eigen::VectorXd(a / vec.array());
      });

  lua.new_usertype<Eigen::RowVectorXd>(
      "row_vector",
      sol::meta_function::to_string,
      [](const Eigen::RowVectorXd& vec) {
        std::stringstream ss;
        ss << vec.transpose();
        return ss.str();
      });

  lua.set_function(
      "log",
      sol::overload(
          [](double a) { return std::log(a); },
          [](const Eigen::VectorXd& vec) {
            return Eigen::VectorXd(vec.array().log());
          },
          [](const Eigen::MatrixXd& mat) {
            return Eigen::MatrixXd(mat.array().log());
          }));

  lua.set_function(
      "log2",
      sol::overload(
          [](double a) { return std::log2(a); },
          [](const Eigen::VectorXd& vec) {
            return Eigen::VectorXd(vec.array().log2());
          },
          [](const Eigen::MatrixXd& mat) {
            return Eigen::MatrixXd(mat.array().log2());
          }));

  lua.set_function(
      "log10",
      sol::overload(
          [](double a) { return std::log10(a); },
          [](const Eigen::VectorXd& vec) {
            return Eigen::VectorXd(vec.array().log10());
          },
          [](const Eigen::MatrixXd& mat) {
            return Eigen::MatrixXd(mat.array().log10());
          }));

  lua.set_function(
      "sqrt",
      sol::overload(
          [](double a) { return std::sqrt(a); },
          [](const Eigen::VectorXd& vec) {
            return Eigen::VectorXd(vec.array().sqrt());
          },
          [](const Eigen::MatrixXd& mat) {
            return Eigen::MatrixXd(mat.array().sqrt());
          }));

  lua.set_function(
      "norm",
      sol::overload(
          [](const Eigen::VectorXd& vec) { return vec.norm(); },
          [](const Eigen::MatrixXd& mat) { return mat.norm(); }));

  lua.set_function(
      "diag",
      sol::overload(
          [](const Eigen::VectorXd& vec) {
            return Eigen::MatrixXd(vec.asDiagonal());
          },
          [](const Eigen::MatrixXd& mat) {
            return Eigen::VectorXd(mat.diagonal());
          }));
}

}  // namespace knoop::lua
