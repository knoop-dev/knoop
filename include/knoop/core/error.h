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

#include <format>
#include <stdexcept>

namespace knoop {

/**
 * \brief Knoop's internal error class, can't be thrown directly
 */
class KnoopError : public std::runtime_error {
 protected:
  template <typename... Args>
  explicit KnoopError(std::format_string<Args...> fmt, Args&&... args)
      : std::runtime_error(std::format(fmt, std::forward<Args>(args)...)) {}
};

/**
 * \brief Generic error type
 */
class KnoopGenericError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopGenericError(std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

/**
 * \brief Thrown when buffer layout does not match what is required by mutator,
 * transformer, or preprocessor.
 */
class KnoopBufferLayoutError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopBufferLayoutError(
      std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

/**
 * \brief Thrown when supplied arguments are invalid, in whatever sense.
 */
class KnoopInvalidArgumentError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopInvalidArgumentError(
      std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

/**
 * \brief Thrown when supplied arguments are outside valid domain or range.
 */
class KnoopRangeError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopRangeError(std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

/**
 * \brief Thrown when result of numerical computations are unstable or the
 * computation fails. Examples include Cholesky factorisation, NaNs in
 * computation, etcetera.
 */
class KnoopNumericalError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopNumericalError(std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

/**
 * \brief Thrown when reading in or writing out data fails..
 */
class KnoopIOError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopIOError(std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

/**
 * \brief Thrown when spousal correlation matrix is infeasible
 */
class KnoopInfeasibleSpouseCorrelationError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopInfeasibleSpouseCorrelationError(
      std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

/**
 * \brief Thrown when pedigree operations are ill posed or fail
 */
class KnoopPedigreeError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopPedigreeError(std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

/**
 * \brief Thrown when dimension of matrix or buffers are incorrect.
 */
class KnoopDimensionError : public KnoopError {
 public:
  template <typename... Args>
  explicit KnoopDimensionError(std::format_string<Args...> fmt, Args&&... args)
      : KnoopError(fmt, std::forward<Args>(args)...) {}
};

}  // namespace knoop
