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
#include <knoop/core/utils.h>

#include <Eigen/Dense>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

namespace knoop {

template <typename T>
inline T parse(const std::string& s);

template <>
inline std::string parse(const std::string& s) {
  return s;
}

template <>
inline double parse(const std::string& s) {
  return std::stod(s);
}

template <typename T>
inline std::vector<T> parseEach(const std::vector<std::string>& ss) {
  std::vector<T> result(ss.size());
  for (std::size_t el = 0; el < ss.size(); ++el) result[el] = parse<T>(ss[el]);
  return result;
}

template <typename T>
inline std::vector<T> parseVector(const std::string& s) {
  return parseEach<T>(utils::splitString(s));
}

template <>
inline std::vector<double> parse(const std::string& s) {
  return parseVector<double>(s);
}

template <>
inline Eigen::VectorXd parse(const std::string& s) {
  std::vector<double> v = parse<std::vector<double>>(s);
  return Eigen::Map<Eigen::VectorXd>(v.data(), v.size());
}

template <>
inline Eigen::MatrixXd parse(const std::string& s) {
  std::string norm = s;
  std::ranges::replace(norm, ';', '\n');
  std::ranges::replace(norm, '\t', ' ');
  std::ranges::replace(norm, ',', ' ');

  std::vector<std::string> row_strs = utils::splitString(norm, '\n');
  if (row_strs.empty())
    throw KnoopInvalidArgumentError("Cannot parse empty string as matrix");

  std::vector<Eigen::VectorXd> rows(row_strs.size());
  for (std::size_t r = 0; r < row_strs.size(); ++r)
    rows[r] = parse<Eigen::VectorXd>(row_strs[r]);

  std::size_t n_rows = rows.size();
  std::size_t n_cols = rows[0].size();
  auto bad = std::ranges::find_if(
      rows, [n_cols](const auto& r) { return r.size() != n_cols; });

  if (bad != rows.end())
    throw KnoopDimensionError(
        "Matrix row {} has {} values, expected {}",
        std::distance(rows.begin(), bad),
        bad->size(),
        n_cols);

  Eigen::MatrixXd matrix(n_rows, n_cols);
  for (std::size_t r = 0; r < n_rows; ++r) matrix.row(r) = rows[r];

  return (n_rows == 1) ? matrix.transpose() : matrix;
}

template <typename T>
inline T parseFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw KnoopIOError("Could not open file {} for parsing", path.string());
  std::string contents(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return parse<T>(contents);
}

inline Eigen::MatrixXd parsePhenoFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw KnoopIOError("Could not open file {} for parsing", path.string());

  std::string line;
  std::getline(file, line);  // skip header

  std::vector<std::vector<double>> matrix;
  while (std::getline(file, line)) {
    std::vector<std::string> tokens = utils::splitString(line, '\t');
    std::vector<double> row;
    for (std::size_t i = 2; i < tokens.size(); ++i)
      row.push_back(std::stod(tokens[i]));
    matrix.push_back(std::move(row));
  }

  if (matrix.empty())
    throw KnoopIOError(
        "Phenotype file {} contains no data rows", path.string());

  std::size_t n_rows = matrix.size();
  std::size_t n_cols = matrix[0].size();
  Eigen::MatrixXd result(n_rows, n_cols);
  for (std::size_t r = 0; r < n_rows; ++r)
    for (std::size_t c = 0; c < n_cols; ++c) result(r, c) = matrix[r][c];
  return result;
}

}  // namespace knoop
