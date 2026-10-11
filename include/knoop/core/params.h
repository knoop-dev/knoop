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
#include <knoop/estimate/estimator.h>
#include <knoop/sample/sample.h>

#include <Eigen/Dense>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace knoop {

struct GenomeParams {
  std::size_t n_loc;
  Eigen::VectorXd locus_freq;
  Eigen::VectorXd locus_rec;
  Eigen::VectorXd locus_mut;
};

struct PhenomeParams {
  std::size_t n_pheno;
  std::vector<std::string> names;
  std::vector<std::size_t> n_locs;
  std::unordered_map<std::string, std::size_t> pheno_ids;

  std::vector<Eigen::VectorXd> pheno_effects;
  std::vector<std::vector<std::size_t>> pheno_loc;

  Eigen::VectorXd var_gen;
  Eigen::VectorXd var_env;
  Eigen::VectorXd var_vert;

  Eigen::MatrixXd gen_cor;
  Eigen::MatrixXd env_cor;

  Eigen::VectorXd rnur_pat;
  Eigen::VectorXd rnur_env;
  Eigen::VectorXd vert_pat;
};

struct MatingParams {
  Eigen::MatrixXd mate_cor;
  double tol;
  std::size_t max_itr;
  double temp_init;
  double temp_decay;
  bool exact;
};

struct EstimatorParams {
  std::vector<PopulationEstimator> population_estimators;
  std::vector<Sample> samples;
};

struct GlobalParams {
  std::size_t n_ind;
  std::size_t n_gens;
  bool pedigree_warmup;
  std::size_t pedigree_max_depth;
  std::uint64_t random_seed;
  std::optional<std::uint64_t> post_init_seed;
  std::filesystem::path out_dir;
  std::size_t rep_id;

  LogLevel log_level;
  bool log_to_file = false;
};

struct Params {
  GenomeParams geno;
  PhenomeParams pheno;
  MatingParams mate;
  EstimatorParams estimate;
  GlobalParams global;
};

}  // namespace knoop
