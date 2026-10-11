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
#include <knoop/core/utils.h>
#include <knoop/io/parse.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

namespace knoop {

template <std::size_t N>
struct FixedString {
  // NOLINTBEGIN(modernize-avoid-c-arrays, google-explicit-constructor)
  char data[N]{};
  constexpr FixedString(const char (&s)[N]) { std::copy_n(s, N, data); }
  // NOLINTEND(modernize-avoid-c-arrays, google-explicit-constructor)
  constexpr bool operator==(const FixedString&) const = default;
};

template <FixedString A, FixedString B>
constexpr bool sameName() {
  if constexpr (std::is_same_v<decltype(A), decltype(B)>)
    return A == B;
  else
    return false;
}

template <FixedString ColumnName, typename ColumnType>
struct Column {
  static constexpr FixedString Name = ColumnName;
  using Type = ColumnType;
};

template <FixedString Name, typename... Columns>
constexpr std::size_t columnIndex() {
  std::size_t result = sizeof...(Columns);
  std::size_t index = 0;
  (void)((sameName<Columns::Name, Name>() ? (result = index, true)
                                          : (++index, false)) ||
         ...);
  return result;
}

template <std::size_t Index, typename... Columns>
using ColumnAtIndex = std::tuple_element_t<Index, std::tuple<Columns...>>;

template <FixedString Name, typename... Columns>
using ColumnAtName = ColumnAtIndex<columnIndex<Name, Columns...>(), Columns...>;

template <typename... Columns>
using Row = std::tuple<typename Columns::Type...>;

template <typename... Columns>
using DataTable = std::tuple<std::vector<typename Columns::Type>...>;

template <typename... Columns>
class Table {
  static_assert(sizeof...(Columns) > 0, "Table must have at least one column");

 public:
  Table() : col_names_{std::string(Columns::Name.data)...} {}

  explicit Table(DataTable<Columns...> table)
      : data_(std::move(table)),
        col_names_{std::string(Columns::Name.data)...} {}

  std::size_t numRows() const { return std::get<0>(data_).size(); }

  static constexpr std::size_t numCols() { return sizeof...(Columns); }

  bool empty() const { return numRows() == 0; }

  const DataTable<Columns...>& data() const { return data_; }

  void readFile(
      const std::filesystem::path& path,
      char col_sep = ',',
      char row_sep = '\n',
      bool has_header = true);

  Row<Columns...> row(std::size_t i) const;
  void insertRow(Row<Columns...> row);
  void insertRows(std::vector<Row<Columns...>> rows);

  Table<Columns...> filter(
      std::function<bool(const Row<Columns...>&)> pred) const;

 private:
  DataTable<Columns...> data_;
  std::array<std::string, sizeof...(Columns)> col_names_;
  std::array<std::size_t, sizeof...(Columns)> col_raw_index_;
  std::ofstream file_;

  void createColumnIndex(const std::string& header, char col_sep);
};

template <typename... Columns>
inline void Table<Columns...>::createColumnIndex(
    const std::string& header, char col_sep) {
  std::vector<std::string> header_cols = utils::splitString(header, col_sep);

  for (std::size_t el = 0; el < col_raw_index_.size(); ++el) {
    auto it = std::ranges::find(header_cols, col_names_[el]);
    if (it == header_cols.end())
      throw std::runtime_error(
          std::format("Column '{}' not found", col_names_[el]));
    col_raw_index_[el] = std::distance(header_cols.begin(), it);
  }
}

template <typename... Columns>
inline void Table<Columns...>::readFile(
    const std::filesystem::path& path,
    char col_sep,
    char row_sep,
    bool has_header) {
  // since we're reading in from a new file, delete current data
  data_ = DataTable<Columns...>{};

  std::string line;
  std::ifstream file(path);

  if (has_header) {
    try {
      std::getline(file, line);
      createColumnIndex(line, col_sep);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::format("{} in file {}", e.what(), path.string()));
    }
  } else {
    std::iota(col_raw_index_.begin(), col_raw_index_.end(), 0);
  }

  while (std::getline(file, line, row_sep)) {
    std::vector<std::string> row_data = utils::splitString(line, col_sep);
    // for each column type, access it at it's prescribed index in the raw data,
    // call parse on it with the parse<T> functionality, and push it to its
    // respective slot in the column index
    (std::get<columnIndex<Columns::Name, Columns...>()>(data_).push_back(
         parse<typename Columns::Type>(
             row_data
                 [col_raw_index_[columnIndex<Columns::Name, Columns...>()]])),
     ...);
  }
}

template <typename... Columns>
inline Row<Columns...> Table<Columns...>::row(std::size_t i) const {
  // tuples are essentially reference containers, so declaration /
  // initialisation must take place inside of the fold expression
  return Row<Columns...>{
      std::get<columnIndex<Columns::Name, Columns...>()>(data_)[i]...};
}

template <typename... Columns>
inline void Table<Columns...>::insertRow(Row<Columns...> row) {
  (std::get<columnIndex<Columns::Name, Columns...>()>(data_).push_back(
       std::get<columnIndex<Columns::Name, Columns...>()>(row)),
   ...);
}

template <typename... Columns>
inline void Table<Columns...>::insertRows(std::vector<Row<Columns...>> rows) {
  for (const auto& row : rows) insertRow(row);
}

template <typename... Columns>
inline Table<Columns...> Table<Columns...>::filter(
    std::function<bool(const Row<Columns...>&)> pred) const {
  std::vector<Row<Columns...>> rows;
  for (std::size_t r = 0; r < numRows(); ++r) {
    Row<Columns...> cur_row = row(r);
    if (pred(cur_row)) rows.push_back(cur_row);
  }

  Table filtered;
  filtered.insertRows(rows);

  return filtered;
}

template <std::size_t Index, typename... Columns>
inline std::vector<typename ColumnAtIndex<Index, Columns...>::Type> getColumn(
    const Table<Columns...>& table) {
  return std::get<Index>(table.data());
}

template <FixedString Name, typename... Columns>
inline std::vector<typename ColumnAtName<Name, Columns...>::Type> getColumn(
    const Table<Columns...>& table) {
  return std::get<columnIndex<Name, Columns...>()>(table.data());
}

template <FixedString... Subset, typename... Columns>
inline Table<ColumnAtName<Subset, Columns...>...> subset(
    const Table<Columns...>& table) {
  return Table<ColumnAtName<Subset, Columns...>...>(
      DataTable<ColumnAtName<Subset, Columns...>...>{
          getColumn<Subset>(table)...});
}

template <FixedString... Subset, typename... Columns>
inline void writeTable(
    const Table<Columns...>& table,
    const std::filesystem::path& path,
    char col_sep = '\t',
    char row_sep = '\n',
    bool include_header = true) {
  if constexpr (sizeof...(Subset) == 0) {
    writeTable<Columns::Name...>(table, path, col_sep, row_sep, include_header);
    return;
  }

  std::ofstream file(path);
  std::array<std::string, sizeof...(Subset)> subset{Subset.data...};

  if (include_header)
    for (std::size_t el = 0; el < subset.size(); ++el)
      file << subset[el] << (el == subset.size() - 1 ? row_sep : col_sep);

  for (std::size_t r = 0; r < table.numRows(); ++r) {
    Row<Columns...> cur_row = table.row(r);

    [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
      ((file << std::format(
            "{}{}",
            std::get<columnIndex<Subset, Columns...>()>(cur_row),
            Indices == sizeof...(Subset) - 1 ? row_sep : col_sep)),
       ...);
    }(std::make_index_sequence<sizeof...(Subset)>{});
  }
}

template <typename... Columns>
class TableXd {
 public:
  std::size_t numRows() const { return known_.numRows(); }

  static constexpr std::size_t numCols() { return sizeof...(Columns); }

  void readFile(
      const std::filesystem::path& path,
      char col_sep = ',',
      char row_sep = '\n',
      bool has_header = true);

  // buf_names, if given, must have one name per matrix() column; overrides
  // whatever readFile() captured (or supplies names when there's nothing to
  // fall back on, e.g. no prior read, or a headerless one). With neither,
  // dynamic columns are written under numeric headers ("1", "2", ...).
  void writeFile(
      const std::filesystem::path& path,
      char col_sep = ',',
      char row_sep = '\n',
      bool include_header = true,
      std::optional<std::vector<std::string>> buf_names = std::nullopt) const;

  const Table<Columns...>& known() const { return known_; }

  const Eigen::MatrixXd& matrix() const { return buf_; }

  // names of the dynamic (matrix()) columns, captured from the file's
  // header by readFile() — empty if the file had no header.
  const std::vector<std::string>& bufNames() const { return buf_names_; }

 private:
  Table<Columns...> known_;
  Eigen::MatrixXd buf_;
  std::vector<std::string> buf_names_;

  // resolves what writeFile() should name the dynamic columns: the
  // explicit argument if given and correctly sized, else buf_names_ (from
  // a prior readFile()), else a plain numeric fallback.
  std::vector<std::string> resolveBufNames(
      std::optional<std::vector<std::string>> buf_names) const;
};

template <typename... Columns>
inline void TableXd<Columns...>::readFile(
    const std::filesystem::path& path,
    char col_sep,
    char row_sep,
    bool has_header) {
  DataTable<Columns...> known_data;
  std::array<std::size_t, sizeof...(Columns)> raw_index;

  std::string line;
  std::ifstream file(path);

  buf_names_.clear();

  if (has_header) {
    std::array<std::string, sizeof...(Columns)> col_names{
        std::string(Columns::Name.data)...};
    std::getline(file, line);
    std::vector<std::string> header_cols = utils::splitString(line, col_sep);
    for (std::size_t el = 0; el < raw_index.size(); ++el) {
      auto it = std::ranges::find(header_cols, col_names[el]);
      if (it == header_cols.end())
        throw std::runtime_error(
            std::format(
                "Column '{}' not found in file {}",
                col_names[el],
                path.string()));
      raw_index[el] = std::distance(header_cols.begin(), it);
    }

    // whatever header text wasn't claimed above becomes the dynamic
    // columns' names, so a later writeFile() can round-trip them.
    for (std::size_t i = 0; i < header_cols.size(); ++i)
      if (std::ranges::find(raw_index, i) == raw_index.end())
        buf_names_.push_back(header_cols[i]);
  } else {
    std::iota(raw_index.begin(), raw_index.end(), 0);
  }

  // buf_text accumulates the row-major, row_sep-joined leftover columns —
  // every raw column not claimed by Columns... — so parse<Eigen::MatrixXd>
  // can build buf_ (and enforce every row being the same width) in one call,
  // rather than re-implementing that check here. Built in the same pass as
  // known_data so the file is only read once.
  std::string buf_text;

  while (std::getline(file, line, row_sep)) {
    std::vector<std::string> row_data = utils::splitString(line, col_sep);

    (std::get<columnIndex<Columns::Name, Columns...>()>(known_data)
         .push_back(
             parse<typename Columns::Type>(
                 row_data
                     [raw_index[columnIndex<Columns::Name, Columns...>()]])),
     ...);

    bool first = true;
    for (std::size_t i = 0; i < row_data.size(); ++i) {
      if (std::ranges::find(raw_index, i) != raw_index.end()) continue;
      if (!first) buf_text += ' ';
      buf_text += row_data[i];
      first = false;
    }
    buf_text += '\n';
  }

  known_ = Table<Columns...>(std::move(known_data));

  buf_ =
      buf_text.empty() ? Eigen::MatrixXd() : parse<Eigen::MatrixXd>(buf_text);
}

template <typename... Columns>
inline std::vector<std::string> TableXd<Columns...>::resolveBufNames(
    std::optional<std::vector<std::string>> buf_names) const {
  if (!buf_names.has_value() && !buf_names_.empty()) buf_names = buf_names_;

  if (buf_names.has_value() &&
      static_cast<Eigen::Index>(buf_names.value().size()) != buf_.cols()) {
    Log::error(
        "TableXd::writeFile: {} names given for {} matrix columns — "
        "falling back to numeric column headers",
        buf_names.value().size(),
        buf_.cols());
    buf_names.reset();
  }

  if (buf_names.has_value()) return buf_names.value();

  std::vector<std::string> numbered;
  numbered.reserve(buf_.cols());
  for (Eigen::Index c = 0; c < buf_.cols(); ++c)
    numbered.push_back(std::to_string(c + 1));

  return numbered;
}

template <typename... Columns>
inline void TableXd<Columns...>::writeFile(
    const std::filesystem::path& path,
    char col_sep,
    char row_sep,
    bool include_header,
    std::optional<std::vector<std::string>> buf_names) const {
  std::vector<std::string> names = resolveBufNames(std::move(buf_names));

  std::ofstream file(path);

  auto join = [&](const std::vector<std::string>& tokens) {
    bool first = true;
    for (const auto& t : tokens) {
      if (!first) file << col_sep;
      file << t;
      first = false;
    }
    file << row_sep;
  };

  if (include_header) {
    std::vector<std::string> header{std::string(Columns::Name.data)...};
    header.insert(header.end(), names.begin(), names.end());
    join(header);
  }

  for (std::size_t r = 0; r < known_.numRows(); ++r) {
    auto known_row = known_.row(r);
    std::vector<std::string> tokens;
    (tokens.push_back(
         std::format(
             "{}",
             std::get<columnIndex<Columns::Name, Columns...>()>(known_row))),
     ...);
    for (Eigen::Index c = 0; c < buf_.cols(); ++c)
      tokens.push_back(std::format("{}", buf_(r, c)));
    join(tokens);
  }
}

/**
 *  Rework Table:
 *
 *   // on the surface level:
 *    Table<std::string, double, double> decl(
 *      Column<std::string>("id") ,
 *      Column<double>("pheno1"),
 *      Column<double>("pheno2"),
 *      Column<std::size_t>("n_loci")
 *    );
 *
 *   // perhaps we abstract like so:
 *    Table decl(
 *      ColumnString("id"),
 *      ColumnDouble("pheno1"),
 *      ColumnDouble("pheno2"),
 *      ColumnInt("n_loci"),
 *      ...
 *    )
 *
 *    decl.readFile("...");
 *
 *    // ...
 */

}  // namespace knoop
