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
#include <knoop/estimate.h>
#include <knoop/io/parse.h>
#include <knoop/sample.h>

#include <Eigen/Dense>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace knoop {

namespace details {

inline Eigen::VectorXd expand(
    const std::variant<double, File<Eigen::MatrixXd>, Distribution>& val,
    std::size_t size) {
  if (std::holds_alternative<double>(val))
    return Eigen::VectorXd::Constant(size, std::get<double>(val));

  if (std::holds_alternative<Distribution>(val))
    return std::get<Distribution>(val)(size);

  Eigen::MatrixXd res = std::get<File<Eigen::MatrixXd>>(val).load();

  if (res.size() == 1) return Eigen::VectorXd::Constant(size, res(0));

  if (static_cast<std::size_t>(res.size()) != size)
    throw std::runtime_error(
        "incorrect vector size; got " + std::to_string(res.size()) +
        ", expected " + std::to_string(size));

  return res;
}

inline Eigen::MatrixXd expandMatrix(
    const std::variant<File<Eigen::MatrixXd>, Eigen::MatrixXd>& val) {
  if (std::holds_alternative<Eigen::MatrixXd>(val))
    return std::get<Eigen::MatrixXd>(val);

  return std::get<File<Eigen::MatrixXd>>(val).load();
}

}  // namespace details

struct Genome {
  std::size_t n_loci = 5000;
  std::variant<double, File<Eigen::MatrixXd>, Distribution> locus_rec = 0.5;
  std::variant<double, File<Eigen::MatrixXd>, Distribution> locus_freq = 0.5;
  std::variant<double, File<Eigen::MatrixXd>, Distribution> locus_mut = 0.0;
};

struct Phenotype {
  std::string name;
  std::size_t n_causal_loci;
  std::unordered_map<std::string, std::size_t> ids;

  std::optional<std::variant<double, File<Eigen::MatrixXd>, Distribution>>
      effects;
  std::optional<
      std::variant<File<std::vector<std::size_t>>, std::vector<std::size_t>>>
      causal_loci;

  double var_genetic = 0.5;
  double var_environmental = 0.5;
  double var_vertical = 0.0;

  double vertical_paternal_ratio = 0.5;
  double nurture_paternal_ratio = 0.5;
  double nurture_environmental_ratio = 0.5;
};

struct Mating {
  std::string type = "random";
  std::optional<std::variant<File<Eigen::MatrixXd>, Eigen::MatrixXd>> mate_cor;

  double tolerance = 0.1;
  std::size_t max_iterations = 0;
  double initial_temperature = 1.0;
  double temperature_decay = 0.99;

  bool exact = false;
};

struct SimulationSpec {
  std::size_t n_individuals = 10000;

  Genome genome;
  std::vector<Phenotype> phenotypes;
  std::optional<std::variant<File<Eigen::MatrixXd>, Eigen::MatrixXd>>
      genetic_component_cor;
  std::optional<std::variant<File<Eigen::MatrixXd>, Eigen::MatrixXd>>
      environmental_component_cor;

  Mating mating;

  std::vector<PopulationEstimator> estimators;  // population-wide estimators

  std::vector<SampleSpec> sample_spec;
  std::vector<SampleEstimatorSpec> sample_estimator_spec;

  std::size_t pedigree_max_depth = 1;
  bool pedigree_warmup = false;

  std::size_t n_replicates = 1;
  std::size_t n_workers = 1;
  std::size_t n_threads_per_worker = 1;
  std::size_t n_threads = 1;
  std::size_t n_generations = 15;
  std::filesystem::path output_dir = ".";
  std::optional<std::string> output_name;
  std::optional<std::uint64_t> random_seed;
  bool share_init_state = false;  // start all replicates from the same state

  LogLevel log_level = LogLevel::Info;
  bool log_to_file = true;

  bool delete_tmp = true;
};

inline PhenomeParams buildPhenoParams(const SimulationSpec& spec) {
  std::size_t n_pheno = spec.phenotypes.size();

  std::vector<std::string> names(n_pheno);
  std::vector<std::size_t> n_locs(n_pheno);
  std::unordered_map<std::string, std::size_t> pheno_ids(n_pheno);

  std::vector<Eigen::VectorXd> pheno_effects(n_pheno);
  std::vector<std::vector<std::size_t>> pheno_loc(n_pheno);

  Eigen::VectorXd var_gen(n_pheno);
  Eigen::VectorXd var_env(n_pheno);
  Eigen::VectorXd var_vert(n_pheno);

  Eigen::VectorXd rnur_pat(n_pheno);
  Eigen::VectorXd rnur_env(n_pheno);
  Eigen::VectorXd vert_pat(n_pheno);

  for (std::size_t pheno = 0; pheno < n_pheno; ++pheno) {
    Phenotype pheno_data = spec.phenotypes[pheno];

    if (pheno_data.n_causal_loci == 0)
      pheno_data.n_causal_loci = spec.genome.n_loci / n_pheno;

    names[pheno] = pheno_data.name;
    n_locs[pheno] = pheno_data.n_causal_loci;
    pheno_ids[pheno_data.name] = pheno;

    {
      Eigen::VectorXd raw_effects;
      if (!pheno_data.effects.has_value()) {
        raw_effects =
            RademacherDistribution::generate(pheno_data.n_causal_loci);
      } else if (std::holds_alternative<double>(pheno_data.effects.value())) {
        raw_effects = Eigen::VectorXd::Constant(
            pheno_data.n_causal_loci,
            std::get<double>(pheno_data.effects.value()));
      } else if (
          std::holds_alternative<Distribution>(pheno_data.effects.value())) {
        raw_effects = std::get<Distribution>(pheno_data.effects.value())(
            pheno_data.n_causal_loci);
      } else {
        raw_effects =
            std::get<File<Eigen::MatrixXd>>(pheno_data.effects.value()).load();
      }
      {
        double var_total_tmp = pheno_data.var_genetic +
                               pheno_data.var_environmental +
                               pheno_data.var_vertical;
        pheno_effects[pheno] =
            raw_effects / raw_effects.norm() *
            std::sqrt(pheno_data.var_genetic / var_total_tmp);
      }
    }

    if (pheno_data.causal_loci.has_value() &&
        spec.genetic_component_cor.has_value())
      throw std::runtime_error(
          "specify either exclusively phenotype causal loci or genetic "
          "component correlation");

    if (pheno_data.causal_loci.has_value()) {
      std::vector<std::size_t> loci;

      auto variant = pheno_data.causal_loci.value();

      if (std::holds_alternative<std::vector<std::size_t>>(variant))
        loci = std::get<std::vector<std::size_t>>(variant);
      else
        loci = std::get<File<std::vector<std::size_t>>>(variant).load();

      if (loci.size() != pheno_data.n_causal_loci)
        throw std::runtime_error(
            "phenotype " + pheno_data.name + " expected " +
            std::to_string(pheno_data.n_causal_loci) + " causal loci, got " +
            std::to_string(loci.size()));
      pheno_loc[pheno] = std::move(loci);
    } else
      pheno_loc[pheno].resize(pheno_data.n_causal_loci);

    double var_total = pheno_data.var_genetic + pheno_data.var_environmental +
                       pheno_data.var_vertical;
    if (var_total <= 0.0)
      throw std::runtime_error(
          "phenotype " + pheno_data.name +
          ": variance components must sum to a positive value");
    var_gen(pheno) = pheno_data.var_genetic / var_total;
    var_env(pheno) = pheno_data.var_environmental / var_total;
    var_vert(pheno) = pheno_data.var_vertical / var_total;
    rnur_pat(pheno) = pheno_data.nurture_paternal_ratio;
    rnur_env(pheno) = pheno_data.nurture_environmental_ratio;
    vert_pat(pheno) = pheno_data.vertical_paternal_ratio;
  }

  Eigen::MatrixXd gen_cor =
      spec.genetic_component_cor.has_value()
          ? details::expandMatrix(spec.genetic_component_cor.value())
          : Eigen::MatrixXd::Identity(n_pheno, n_pheno);

  Eigen::MatrixXd env_cor =
      spec.environmental_component_cor.has_value()
          ? details::expandMatrix(spec.environmental_component_cor.value())
          : Eigen::MatrixXd::Identity(n_pheno, n_pheno);

  return PhenomeParams{
      .n_pheno = n_pheno,
      .names = std::move(names),
      .n_locs = std::move(n_locs),
      .pheno_ids = std::move(pheno_ids),
      .pheno_effects = std::move(pheno_effects),
      .pheno_loc = std::move(pheno_loc),
      .var_gen = var_gen,
      .var_env = var_env,
      .var_vert = var_vert,
      .gen_cor = gen_cor,
      .env_cor = env_cor,
      .rnur_pat = rnur_pat,
      .rnur_env = rnur_env,
      .vert_pat = vert_pat};
}

inline Params buildParams(const SimulationSpec& spec) {
  GenomeParams geno = GenomeParams{
      .n_loc = spec.genome.n_loci,
      .locus_freq = details::expand(spec.genome.locus_freq, spec.genome.n_loci),
      .locus_rec = details::expand(spec.genome.locus_rec, spec.genome.n_loci),
      .locus_mut = details::expand(spec.genome.locus_mut, spec.genome.n_loci)};

  PhenomeParams pheno = buildPhenoParams(spec);

  Eigen::MatrixXd mate_cor =
      (spec.mating.mate_cor.has_value())
          ? details::expandMatrix(spec.mating.mate_cor.value())
          : Eigen::MatrixXd::Zero(pheno.n_pheno, pheno.n_pheno);

  MatingParams mate = MatingParams{
      .mate_cor = std::move(mate_cor),
      .tol = spec.mating.tolerance,
      .max_itr = spec.mating.max_iterations,
      .temp_init = spec.mating.initial_temperature,
      .temp_decay = spec.mating.temperature_decay};

  EstimatorParams estimate = EstimatorParams{
      .population_estimators = spec.estimators,
      .samples = buildSamples(spec.sample_spec, spec.sample_estimator_spec)};

  GlobalParams sim = GlobalParams{
      .n_ind = spec.n_individuals,
      .n_gens = spec.n_generations,
      .pedigree_warmup = spec.pedigree_warmup,
      .pedigree_max_depth = spec.pedigree_max_depth,
      .random_seed = spec.random_seed.value_or(0),
      .out_dir = spec.output_dir,
      .log_level = spec.log_level,
      .log_to_file = spec.log_to_file};

  return Params{
      .geno = std::move(geno),
      .pheno = std::move(pheno),
      .mate = std::move(mate),
      .estimate = std::move(estimate),
      .global = std::move(sim)};
}

}  // namespace knoop
