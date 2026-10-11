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
#include <knoop/estimate/population_factory.h>
#include <knoop/init/setup.h>
#include <toml++/toml.h>

#include <concepts>
#include <cstdint>
#include <fstream>

namespace knoop {

class WriteGuard {
 public:
  WriteGuard(toml::table*& target, toml::table& scope)
      : target_(target), previous_(target) {
    target_ = &scope;
  }

  ~WriteGuard() { target_ = previous_; }
  WriteGuard(const WriteGuard&) = delete;
  WriteGuard& operator=(const WriteGuard&) = delete;

 private:
  toml::table*& target_;
  toml::table* previous_;
};

class ConfigWriter {
 public:
  explicit ConfigWriter(const SimulationSpec& spec)
      : config_(
            toml::table{
                {"global", toml::table{}},
                {"genome", toml::table{}},
                {"phenotypes", toml::table{}},
                {"mating", toml::table{}},
                {"estimators", toml::table{}},
                {"samples", toml::table{}}}),
        spec_(spec),
        write_to_(config_["global"].as_table()) {
    std::filesystem::path config_path =
        spec.output_name.has_value()
            ? spec.output_dir / ("config_" + spec.output_name.value() + ".toml")
            : spec.output_dir / "config.toml";

    writeConfig(config_path);
  }

 private:
  toml::table config_;
  toml::table* write_to_;
  const SimulationSpec& spec_;

  void writeToSection(const std::string& name);

  template <typename T>
  auto toTOML(const T& val);

  template <std::unsigned_integral T>
  auto toTOML(const T& val);

  template <typename T>
  auto toTOML(const File<T>& val);

  template <typename T>
  auto toTOML(const std::vector<T>& val);

  template <typename T>
  toml::array toArrayTOML(const std::vector<T>& val);

  template <typename T>
  toml::table toTableTOML(const std::vector<T>& val);

  template <typename T>
  void writeParam(const T& val, const std::string& name);

  template <typename T>
  void writeParam(const std::optional<T>& val, const std::string& name);

  template <typename T>
  void writeParam(const std::vector<T>& val, const std::string& name);

  template <typename... Ts>
  void writeParam(const std::variant<Ts...>& val, const std::string& name);

  void writeGlobalConfig();
  void writeGenomeConfig();
  void writePhenotypeConfig();
  void writeMatingConfig();
  void writeEstimatorConfig();
  void writeSampleConfig();
  void writeConfig(const std::filesystem::path& config_path);
};

inline void ConfigWriter::writeToSection(const std::string& name) {
  write_to_ = config_[name].as_table();
}

template <typename T>
inline auto ConfigWriter::toTOML(const T& val) {
  static_assert(
      toml::impl::is_native<T> || toml::is_container<T>,
      "toTOML: no conversion known for this type — add specialisation");
  return val;
}

template <typename T>
inline auto ConfigWriter::toTOML(const File<T>& val) {
  return "file(" + val.path.string() + ")";
}

template <typename T>
inline auto ConfigWriter::toTOML(const std::vector<T>& val) {
  return toArrayTOML(val);
}

template <std::unsigned_integral T>
inline auto ConfigWriter::toTOML(const T& val) {
  return static_cast<int64_t>(val);
}

template <>
inline auto ConfigWriter::toTOML(const std::filesystem::path& val) {
  return val.string();
}

template <>
inline auto ConfigWriter::toTOML(const LogLevel& val) {
  return logLevelToString(val);
}

template <>
inline auto ConfigWriter::toTOML(const Distribution& val) {
  return val.name + "(" + utils::vectorToString(val.params) + ")";
}

template <>
inline auto ConfigWriter::toTOML(const Eigen::MatrixXd& val) {
  toml::array arr;

  for (std::size_t row = 0; row < val.rows(); ++row) {
    arr.push_back(toml::array{});
    auto* row_arr = arr.back().as_array();
    for (std::size_t col = 0; col < val.cols(); ++col)
      row_arr->push_back(val(row, col));
  }

  return arr;
}

template <>
inline auto ConfigWriter::toTOML(const PopulationEstimator& val) {
  return val.name;
}

template <>
inline auto ConfigWriter::toTOML(const Phenotype& val) {
  toml::table phenotype;

  {
    WriteGuard write(write_to_, phenotype);
    writeParam(val.n_causal_loci, "n_loci");
    writeParam(val.effects, "effects");
    writeParam(val.causal_loci, "causal_loci");
    writeParam(val.var_genetic, "var_genetic");
    writeParam(val.var_environmental, "var_environmental");
    writeParam(val.var_vertical, "var_vertical");
  }

  return phenotype;
}

template <>
inline auto ConfigWriter::toTOML(const SampleSpec& val) {
  toml::table sample;

  {
    WriteGuard write(write_to_, sample);
    writeParam(val.proband_type, "proband_type");
    writeParam(val.n_probands, "n_probands");
    writeParam(val.on, "on");
    writeParam(val.of, "of");
    writeParam(val.include, "include");
    writeParam(val.weight_function, "weight_function");
    writeParam(val.agg, "agg");
    writeParam(val.estimators, "estimators");
  }

  return sample;
}

template <>
inline auto ConfigWriter::toTOML(const SampleEstimatorSpec& val) {
  toml::table sample_estimator;

  {
    WriteGuard write(write_to_, sample_estimator);
    writeParam(val.type, "type");
    writeParam(val.params, "params");
    writeParam(val.exec, "exec");
    writeParam(val.exec_args, "exec_args");
    writeParam(val.n_rows, "n_rows");
    writeParam(val.n_cols, "n_cols");
    writeParam(val.row_names, "row_names");
    writeParam(val.col_names, "col_names");
  }

  return sample_estimator;
}

template <typename T>
inline toml::array ConfigWriter::toArrayTOML(const std::vector<T>& val) {
  toml::array arr{};
  for (std::size_t el = 0; el < val.size(); ++el)
    arr.push_back(toTOML(val[el]));
  return arr;
}

template <typename T>
inline toml::table ConfigWriter::toTableTOML(const std::vector<T>& val) {
  toml::table tbl{};
  for (const auto& item : val) tbl.insert(item.name, toTOML(item));
  return tbl;
}

template <typename T>
inline void ConfigWriter::writeParam(const T& val, const std::string& name) {
  write_to_->insert(name, toTOML(val));
}

template <typename T>
inline void ConfigWriter::writeParam(
    const std::optional<T>& val, const std::string& name) {
  if (val.has_value()) writeParam(val.value(), name);
}

template <typename T>
inline void ConfigWriter::writeParam(
    const std::vector<T>& val, const std::string& name) {
  if constexpr (
      std::is_same_v<decltype(toTOML(std::declval<T>())), toml::table>)
    write_to_->insert(name, toTableTOML(val));
  else
    write_to_->insert(name, toArrayTOML(val));
}

template <typename... Ts>
inline void ConfigWriter::writeParam(
    const std::variant<Ts...>& val, const std::string& name) {
  std::visit([this, name](const auto& v) { writeParam(v, name); }, val);
}

inline void ConfigWriter::writeGlobalConfig() {
  writeToSection("global");
  writeParam(spec_.n_individuals, "n_individuals");
  writeParam(spec_.n_generations, "n_generations");
  writeParam(spec_.n_replicates, "n_replicates");
  writeParam(spec_.n_workers, "n_workers");
  writeParam(spec_.n_threads_per_worker, "n_threads_per_worker");
  writeParam(spec_.n_threads, "n_threads");
  writeParam(spec_.random_seed, "random_seed");
  writeParam(spec_.output_dir, "output_dir");
  writeParam(spec_.output_name, "output_name");
  writeParam(spec_.log_level, "log_level");
  writeParam(spec_.log_to_file, "log_to_file");
  writeParam(spec_.share_init_state, "share_init_state");
  writeParam(spec_.pedigree_max_depth, "pedigree_max_depth");
  writeParam(spec_.pedigree_warmup, "pedigree_warmup");
}

inline void ConfigWriter::writeGenomeConfig() {
  writeToSection("genome");
  writeParam(spec_.genome.n_loci, "n_loci");
  writeParam(spec_.genome.locus_freq, "locus_maf");
  writeParam(spec_.genome.locus_rec, "locus_rec");
  writeParam(spec_.genome.locus_mut, "locus_mut");
}

inline void ConfigWriter::writePhenotypeConfig() {
  writeToSection("phenotypes");
  writeParam(spec_.genetic_component_cor, "genetic_cor");
  writeParam(spec_.environmental_component_cor, "environmental_cor");
  writeParam(spec_.phenotypes, "phenotype");
}

inline void ConfigWriter::writeMatingConfig() {
  writeToSection("mating");
  writeParam(spec_.mating.type, "type");

  if (spec_.mating.type == "assortative") {
    if (spec_.mating.mate_cor.has_value())
      writeParam(spec_.mating.mate_cor.value(), "mate_cor");

    writeParam(spec_.mating.tolerance, "tol_inf");
    writeParam(spec_.mating.max_iterations, "max_itr");
    writeParam(spec_.mating.initial_temperature, "temp_init");
    writeParam(spec_.mating.temperature_decay, "temp_decay");
    writeParam(spec_.mating.exact, "exact");
  }
}

inline void ConfigWriter::writeEstimatorConfig() {
  writeToSection("estimators");
  writeParam(spec_.estimators, "population");
}

inline void ConfigWriter::writeSampleConfig() {
  writeToSection("samples");
  writeParam(spec_.sample_spec, "sample");
  writeParam(spec_.sample_estimator_spec, "sample_estimators");
}

inline void ConfigWriter::writeConfig(const std::filesystem::path& path) {
  writeGlobalConfig();
  writeGenomeConfig();
  writePhenotypeConfig();
  writeMatingConfig();
  writeEstimatorConfig();
  writeSampleConfig();

  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Could not open configuration file for writing");

  out << toml::
             toml_formatter{config_, toml::format_flags::relaxed_float_precision}
      << std::endl;
}

class ConfigReader {
 public:
  explicit ConfigReader(const std::filesystem::path& config_path)
      : config_(toml::parse_file(config_path.string())) {
    readConfig();
  }

  SimulationSpec& result() { return spec_; }

 private:
  toml::table config_;
  SimulationSpec spec_;
  toml::node_view<toml::node> visit(const std::string& path);

  template <typename T>
  T readParam(const std::string& read_from);

  template <typename T>
  void readParam(T& read_to, const std::string& read_from);

  template <typename T>
  void readParam(
      std::variant<File<T>, T>& read_to, const std::string& read_from);

  template <typename T>
  void readParam(std::optional<T>& read_to, const std::string& read_from);

  void readGlobalConfig();
  void readGenomeConfig();
  void readPhenotypeConfig();
  void readMatingConfig();
  void readEstimatorConfig();
  void readSampleConfig();
  void readConfig();
};

inline toml::node_view<toml::node> ConfigReader::visit(
    const std::string& path) {
  std::vector<std::string> tokens = utils::splitString(path, '/');
  toml::node_view<toml::node> node{config_};
  for (const auto& token : tokens) node = node[token];
  return node;
}

template <typename T>
inline T ConfigReader::readParam(const std::string& read_from) {
  T result{};
  toml::node_view<toml::node> node = visit(read_from);
  if (auto val = node.value<T>()) result = *val;
  return result;
}

template <typename T>
inline void ConfigReader::readParam(T& read_to, const std::string& read_from) {
  read_to = readParam<T>(read_from);
}

template <typename T>
inline void ConfigReader::readParam(
    std::variant<File<T>, T>& read_to, const std::string& read_from) {
  toml::node_view<toml::node> node = visit(read_from);

  if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
    if (auto* arr = node.as_array()) {
      auto rows = arr->size();
      auto cols = arr->at(0).as_array()->size();
      Eigen::MatrixXd matrix(rows, cols);

      for (std::size_t r = 0; r < rows; ++r) {
        auto* row = arr->at(r).as_array();
        for (std::size_t c = 0; c < cols; ++c)
          matrix(r, c) = row->at(c).value<double>().value();
      }
      read_to = matrix;
    } else if (const auto& val = node.value<std::string>()) {
      auto [file_tag, file_params] = parseFunction(*val);
      read_to = parse<Eigen::MatrixXd>(file_params[0]);
    }
  } else if constexpr (std::is_same_v<T, std::vector<std::size_t>>) {
    if (auto* arr = node.as_array()) {
      auto n_elem = arr->size();
      std::vector<std::size_t> vector(n_elem);

      for (std::size_t el = 0; el < n_elem; ++el)
        vector[el] = arr->at(el).value<std::size_t>().value();

      read_to = vector;
    } else if (const auto& val = node.value<std::string>()) {
      auto [file_tag, file_params] = parseFunction(*val);
      read_to = parseFile<std::vector<std::size_t>>(file_params[0]);
    }
  } else {
    throw std::invalid_argument(
        "Invalid argument supplied to ConfigReader::readParam for file / "
        "container parsing");
  }
}

template <>
inline void ConfigReader::readParam(
    std::variant<double, File<Eigen::MatrixXd>, Distribution>& read_to,
    const std::string& read_from) {
  toml::node_view<toml::node> node = visit(read_from);

  if (auto val = node.value<double>()) {
    read_to = *val;
  } else if (const auto& val = node.value<std::string>()) {
    auto [fn_name, params] = parseFunction(*val);

    if (fn_name == "file")
      read_to = File<Eigen::MatrixXd>{params[0]};
    else
      read_to = parse<Distribution>(*val);
  } else {
    throw std::runtime_error(
        "Unrecognised type; expecting functional or double");
  }
}

template <>
inline void ConfigReader::readParam(
    std::filesystem::path& read_to, const std::string& read_from) {
  std::string string_path;
  readParam<std::string>(string_path, read_from);
  read_to = static_cast<std::filesystem::path>(string_path);
}

template <>
inline void ConfigReader::readParam(
    LogLevel& read_to, const std::string& read_from) {
  std::string string_log_level;
  readParam<std::string>(string_log_level, read_from);
  read_to = logLevelFromString(string_log_level);
}

template <>
inline std::vector<std::string> ConfigReader::readParam(
    const std::string& read_from) {
  std::vector<std::string> result;
  auto* arr = visit(read_from).as_array();
  if (!arr) return result;

  result.reserve(arr->size());
  for (auto&& el : *arr) result.push_back(el.value<std::string>().value());
  return result;
}

template <typename T>
inline void ConfigReader::readParam(
    std::optional<T>& read_to, const std::string& read_from) {
  toml::node_view<toml::node> node = visit(read_from);

  if (!node) {
    read_to = std::nullopt;
    return;
  }

  read_to.emplace();
  readParam(read_to.value(), read_from);
}

inline void ConfigReader::readGlobalConfig() {
  readParam(spec_.n_individuals, "global/n_individuals");
  readParam(spec_.n_generations, "global/n_generations");
  readParam(spec_.n_replicates, "global/n_replicates");
  readParam(spec_.n_workers, "global/n_workers");
  readParam(spec_.n_threads_per_worker, "global/n_threads_per_worker");
  readParam(spec_.n_threads, "global/n_threads");
  readParam(spec_.pedigree_max_depth, "global/pedigree_max_depth");
  readParam(spec_.pedigree_warmup, "global/pedigree_warmup");
  readParam(spec_.random_seed, "global/random_seed");
  readParam(spec_.output_dir, "global/output_dir");
  readParam(spec_.output_name, "global/output_name");
  readParam(spec_.log_level, "global/log_level");
  readParam(spec_.log_to_file, "global/log_to_file");
  readParam(spec_.share_init_state, "global/share_init_state");
}

inline void ConfigReader::readGenomeConfig() {
  readParam(spec_.genome.n_loci, "genome/n_loci");
  readParam(spec_.genome.locus_freq, "genome/locus_maf");
  readParam(spec_.genome.locus_mut, "genome/locus_mut");
  readParam(spec_.genome.locus_rec, "genome/locus_rec");
}

inline void ConfigReader::readPhenotypeConfig() {
  const auto& phenotypes = config_["phenotypes"]["phenotype"].as_table();

  for (auto&& [pheno_name, pheno_spec] : *phenotypes) {
    std::string pheno_path =
        std::format("phenotypes/phenotype/{}", pheno_name.str());
    auto& pheno = *pheno_spec.as_table();
    Phenotype phenotype;

    phenotype.name = pheno_name.str();
    readParam(phenotype.n_causal_loci, pheno_path + "/n_loci");
    readParam(phenotype.effects, pheno_path + "/effects");
    readParam(phenotype.causal_loci, pheno_path + "/causal_loci");
    readParam(phenotype.var_genetic, pheno_path + "/var_genetic");
    readParam(phenotype.var_environmental, pheno_path + "/var_environmental");
    readParam(phenotype.var_vertical, pheno_path + "/var_vertical");

    spec_.phenotypes.push_back(phenotype);
  }

  readParam(spec_.genetic_component_cor, "phenotypes/genetic_cor");
  readParam(spec_.environmental_component_cor, "phenotypes/environmental_cor");
}

inline void ConfigReader::readMatingConfig() {
  readParam(spec_.mating.type, "mating/type");
  if (spec_.mating.type == "assortative") {
    readParam(spec_.mating.mate_cor, "mating/mate_cor");
    readParam(spec_.mating.tolerance, "mating/tol_inf");
    readParam(spec_.mating.max_iterations, "mating/max_itr");
    readParam(spec_.mating.initial_temperature, "mating/temp_init");
    readParam(spec_.mating.temperature_decay, "mating/temp_decay");
    readParam(spec_.mating.exact, "mating/exact");
  }
}

inline void ConfigReader::readEstimatorConfig() {
  const auto& estimators = config_["estimators"]["population"].as_array();

  for (auto&& estimator : *estimators) {
    std::string raw = *estimator.value<std::string>();
    auto [name, params] = parseFunction(raw);
    PopulationEstimator est = buildPopulationEstimator(name, params);
    est.name = raw;
    spec_.estimators.push_back(std::move(est));
  }
}

inline void ConfigReader::readSampleConfig() {
  if (const auto* sample_estimators_tbl =
          config_["samples"]["sample_estimators"].as_table()) {
    for (auto&& [est_name, est_node] : *sample_estimators_tbl) {
      std::string est_path =
          std::format("samples/sample_estimators/{}", est_name.str());

      SampleEstimatorSpec est_spec;
      est_spec.name = est_name.str();
      readParam(est_spec.type, est_path + "/type");
      readParam(est_spec.params, est_path + "/params");
      readParam(est_spec.exec, est_path + "/exec");
      readParam(est_spec.exec_args, est_path + "/exec_args");
      readParam(est_spec.n_rows, est_path + "/n_rows");
      readParam(est_spec.n_cols, est_path + "/n_cols");
      readParam(est_spec.row_names, est_path + "/row_names");
      readParam(est_spec.col_names, est_path + "/col_names");

      spec_.sample_estimator_spec.push_back(std::move(est_spec));
    }
  }

  if (const auto* samples_tbl = config_["samples"]["sample"].as_table()) {
    for (auto&& [s_name, s_node] : *samples_tbl) {
      std::string s_path = std::format("samples/sample/{}", s_name.str());

      SampleSpec s_spec;
      s_spec.name = s_name.str();
      readParam(s_spec.proband_type, s_path + "/proband_type");
      readParam(s_spec.n_probands, s_path + "/n_probands");
      readParam(s_spec.on, s_path + "/on");
      readParam(s_spec.of, s_path + "/of");
      readParam(s_spec.include, s_path + "/include");
      readParam(s_spec.weight_function, s_path + "/weight_function");
      readParam(s_spec.agg, s_path + "/agg");
      readParam(s_spec.estimators, s_path + "/estimators");

      spec_.sample_spec.push_back(std::move(s_spec));
    }
  }
}

inline void ConfigReader::readConfig() {
  readGlobalConfig();
  readGenomeConfig();
  readPhenotypeConfig();
  readMatingConfig();
  readEstimatorConfig();
  readSampleConfig();
}

}  // namespace knoop
