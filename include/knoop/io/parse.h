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

#include <knoop/core/distributions.h>
#include <knoop/core/utils.h>
#include <knoop/sample/proband.h>
#include <knoop/sample/selection.h>

#include <Eigen/Dense>
#include <algorithm>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

namespace knoop {

inline std::string parseExceptionStr(
    const std::string& s, const std::optional<std::string>& flag) {
  return std::format(
      "Failed to parse '{}' {}",
      s,
      flag.has_value() ? "(passed to " + flag.value() + ")" : "");
}

// TODO: add templating to specify the arguments to be parsed from the function.
// For example, when parsing case-control weight functions, we would like to be
// able to declare three parameters: a threshold (double), a direction
// ('<'|'>'), and a vector of weights (Eigen::VectorXd).
//
// So we could, for example, while processing the params (which is currently
// done via utils::splitString), we could declare
//
//   std::vector<std::string> params = parseParams<
//      Param<"threshold", double, 1>,
//      Param<"direction", std::string, 1>,
//      Param<"effects", Eigen::VectorXd, -1>>(param_str);
//
// Param could be defined as follows:
//
//   template <FixedString ParamName, typename ParamType, int ParamSize>
//   struct Param {
//     static constexpr auto Name = ParamName;
//     using Type = ParamType;
//     static constexpr Size = ParamSize;
//   }
//
// The struct fields Type and Size refer to the type supplied to the parse
// function parse<Type>, and Size refers to the number of elements that the
// parameter should contain. For example, if Param::Type is a vector of integers
// with Param::Size == 2, then we parse two integers into a std::vector<int>.
inline std::pair<std::string, std::vector<std::string>> parseFunction(
    const std::string& s) {
  int paren_begin = s.find('(');
  int paren_end = s.find(')');

  // this is the case where f is used to denote f()
  if (paren_begin == std::string::npos || paren_end == std::string::npos)
    return {s, {}};

  // broken string
  if (paren_begin >= paren_end)
    throw std::runtime_error("Invalid functional form: " + s);

  std::string fn_name = s.substr(0, paren_begin);
  std::string params_str =
      s.substr(paren_begin + 1, paren_end - paren_begin - 1);
  std::vector<std::string> params = utils::splitString(params_str, ',');

  boost::to_lower(fn_name);

  return {fn_name, params};
}

template <typename T>
inline T parse(const std::string& s);

template <>
inline std::string parse(const std::string& s) {
  return s;
}

template <std::unsigned_integral T>
inline T parse(const std::string& s) {
  return static_cast<T>(std::stoull(s));
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

template <typename T>
inline T parse(const std::string& s, const std::optional<std::string>& flag) {
  try {
    return parse<T>(s);
  } catch (const std::exception& e) {
    throw std::runtime_error(parseExceptionStr(s, flag));
  }
}

template <>
inline std::vector<std::size_t> parse(const std::string& s) {
  return parseVector<std::size_t>(s);
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
  std::vector<Eigen::VectorXd> rows(row_strs.size());
  for (std::size_t r = 0; r < row_strs.size(); ++r)
    rows[r] = parse<Eigen::VectorXd>(row_strs[r]);

  std::size_t n_rows = rows.size();
  std::size_t n_cols = rows[0].size();
  auto bad = std::ranges::find_if(
      rows, [n_cols](const auto& r) { return r.size() != n_cols; });

  if (bad != rows.end())
    throw std::runtime_error(
        std::format(
            "Matrix row {} has {} values, expected {}",
            std::distance(rows.begin(), bad),
            bad->size(),
            n_cols));

  Eigen::MatrixXd matrix(n_rows, n_cols);

  for (std::size_t r = 0; r < rows.size(); ++r) {
    if (static_cast<std::size_t>(rows[r].size()) != n_cols)
      throw std::runtime_error(
          std::format(
              "Matrix row {} has {} values, expected {}",
              r,
              rows[r].size(),
              n_cols));
    matrix.row(r) = rows[r];
  }

  return (rows.size() == 1) ? matrix.transpose() : matrix;
}

template <>
inline Distribution parse(const std::string& s) {
  auto [dist_name, dist_params_str] = parseFunction(s);
  std::vector<double> dist_params(dist_params_str.size());

  if (!dist_params_str.empty())
    dist_params = parseEach<double>(dist_params_str);

  return knoop::strToDistribution(dist_name, dist_params);
}

template <>
inline WeightFunction parse(const std::string& s) {
  auto [weight_name, params_str] = parseFunction(s);

  if (weight_name == "uniform") return knoop::uniform();
  if (weight_name == "logistic") {
    auto params = parseEach<double>(params_str);
    return knoop::logistic(
        Eigen::Map<Eigen::VectorXd>(params.data(), params.size()));
  }
  if (weight_name == "case-control") {
    if (params_str.size() < 2)
      throw std::runtime_error(
          "case-control requires at least threshold and direction (< or >)");

    double threshold = parse<double>(params_str[0]);

    std::string dir_token = params_str[1];
    boost::trim(dir_token);
    bool above;
    if (dir_token == ">")
      above = true;
    else if (dir_token == "<")
      above = false;
    else
      throw std::runtime_error("case-control direction must be '<' or '>'");

    std::vector<double> effects = parseEach<double>(
        std::vector<std::string>(params_str.begin() + 2, params_str.end()));

    return knoop::caseControl(
        threshold,
        above,
        Eigen::Map<Eigen::VectorXd>(effects.data(), effects.size()));
  }

  throw std::runtime_error("Unrecognised weight function " + s);
}

template <ProbandType P>
inline ProbandMemberEnum<P> parse(const std::string& s) {
  std::vector<std::string> probands_str = utils::splitString(s);
  std::vector<ProbandMemberEnum<P>> probands(probands_str.size());

  if (probands.empty())
    throw std::runtime_error("Empty proband string supplied");

  std::ranges::transform(
      probands_str, probands.begin(), [](const std::string& s) {
        return probandEnumFromString<P>(s);
      });

  // static cast required since the accumulation operation produces
  // ProbandEnumType<P> but return type is ProbandEnum<P>
  return static_cast<ProbandMemberEnum<P>>(std::accumulate(
      probands.begin() + 1,
      probands.end(),
      probands[0],
      [](const ProbandMemberEnum<P>& a, ProbandMemberEnum<P> b) {
        return a | b;
      }));
}

template <typename T>
inline std::vector<T> parseVector(
    const std::string& s, const std::optional<std::string>& flag) {
  try {
    return parseVector<T>(s);
  } catch (const std::exception& e) {
    throw std::runtime_error(parseExceptionStr(s, flag));
  }
}

template <typename T>
inline T parseFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error(
        "Could not open file " + path.string() + " for parsing");
  std::string contents(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return parse<T>(contents);
}

template <typename T>
struct File {
  std::filesystem::path path;
  T load() const { return parseFile<T>(path); }
};

// TODO: deprecate this in favour of TableXd imports
inline Eigen::MatrixXd parsePhenoFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error(
        "Could not open file " + path.string() + " for parsing");

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

  std::size_t n_rows = matrix.size();
  std::size_t n_cols = matrix[0].size();
  Eigen::MatrixXd result(n_rows, n_cols);
  for (std::size_t r = 0; r < n_rows; ++r)
    for (std::size_t c = 0; c < n_cols; ++c) result(r, c) = matrix[r][c];
  return result;
}

}  // namespace knoop
