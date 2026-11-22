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
#include <knoop/init.h>
#include <knoop/mutate.h>
#include <knoop/sample.h>
#include <omp.h>

#include <chrono>
#include <iostream>

namespace knoop {

namespace details {

inline std::uint64_t shuffleSeed(std::uint64_t rng_seed, std::size_t rep_id) {
  constexpr std::uint64_t kSeedStride = 0x9E3779B97F4A7C15ULL;
  return rng_seed + (kSeedStride * rep_id);
}

inline void preprocessSimulation(Params& params) {
  Log::info("Preprocessing simulation");

  InducePleiotropy pleiotropy(params);
  pleiotropy();
}

inline std::filesystem::path outputSetup(std::uint64_t seed) {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                              std::format("knoop_{}_{}", seed, ms);
  std::filesystem::create_directories(tmp);
  Log::debug("Created simulation workspace directory {}", tmp.string());
  return tmp;
}

inline std::string resultsFilename(const Params& params) {
  return params.outputName().has_value()
             ? std::format("results_{}.h5", params.outputName().value())
             : "results.h5";
}

inline std::string logFilename(const Params& params) {
  return params.outputName().has_value()
             ? std::format("knoop_{}.log", params.outputName().value())
             : "knoop.log";
}

inline void logSetup(const Params& params) {
  if (params.logToFile()) {
    std::filesystem::path log_path =
        params.outputDirectory() / logFilename(params);
    Log::file(log_path, params.logLevel());
  } else {
    Log::stream(std::cout, params.logLevel());
  }
}

inline std::filesystem::path replicateSetup(
    const std::filesystem::path& tmp_dir, std::size_t rep_id) {
  return tmp_dir / std::format("rep_{:03}", rep_id + 1);
}

inline void writerSetup(const Params& params) {
  std::filesystem::path results_path =
      params.outputDirectory() / resultsFilename(params);
  HDF5Writer::getInstance(
      results_path, params.numGenerations(), params.numReplicates());
}

}  // namespace details

inline void runReplicate(const Params& params) {
  boost::random::xoshiro256pp& engine = random::Xoshiro256pp::getInstance();
  random::Xoshiro256pp::setSeed(engine, params.randomSeed());

  if (!std::filesystem::exists(params.outputDirectory()))
    std::filesystem::create_directory(params.outputDirectory());

  State state = buildState(params);

  ComputePopulationEstimates population_est(params);
  ComputeSampleEstimates sample_est(params);
  GenerateHaplotypes haplo(params);
  RandomMating founder_mate(params);
  ScorePhenotypes score(params);
  AssortativeMating mate(params);
  Inherit update(params);

  // initial state generation
  haplo(state);
  state.geno().computeLocusStats();

  // if pedigree_warmup is true, then we simulate pedigree_max_depth
  // founder generations, otherwise just the one
  std::size_t n_it =
      params.pedigreeWarmup() ? params.pedigreeMaxDepth() : 1;

  for (std::size_t it = 0; it < n_it; ++it) {
    Log::info("Simulating founder generation {}", it + 1);
    state.geno().computeLocusStats();
    score(state);
    state.pheno().computeStats();
    founder_mate(state);
    state.updatePedigree();
    if (it == n_it - 1) {
      state.gen = 0;
      population_est(state);
      sample_est(state);
    }
    state.transpose();
    update(state);
    state.transpose();
    state.advance();
  }

  Log::info("Finished simulating founder generation(s)");

  if (params.postInitSeed().has_value()) {
    random::Xoshiro256pp::setSeed(params.postInitSeed());
    Log::debug(
        "Setting post init seed from shared state to {}",
        params.postInitSeed().value());
  }

  HDF5Writer::create(
      "mate-objective",
      utils::vectorPrefix(params.pheno.names(), "male_"),
      utils::vectorPrefix(params.pheno.names(), "female_"),
      params.pheno.size(),
      params.pheno.size());

  // run the core simulation loop
  while (state.gen < params.numGenerations()) {
    Log::info("Simulating generation {}", state.gen);
    state.geno().computeLocusStats();

    // score phenotypes
    score(state);
    state.pheno().computeStats();

    // match mates
    if (params.mate.type() == MatingType::Assortative)
      mate(state);
    else
      founder_mate(state);

    state.updatePedigree();

    HDF5Writer::write(mate.objective(), "mate-objective", state.rep, state.gen);

    // sample and estimate subpopulations
    sample_est(state);

    // compute population estimates
    population_est(state);

    // update the genome — individual-major
    state.transpose();
    update(state);
    state.transpose();

    state.advance();
  }
}

inline void runSimulation(Params& params) {
  // configure OpenMP
  if (params.numWorkers() > 1 && params.numThreadsPerWorker() > 1)
    omp_set_max_active_levels(2);

  // set up the log
  details::logSetup(params);

  // set up random seed
  std::uint64_t seed = random::Xoshiro256pp::setSeed(params.randomSeed());

  /** preprocess */
  details::preprocessSimulation(params);
  params.randomSeed() = seed;

  // validate output directory
  if (!std::filesystem::exists(params.outputDirectory())) {
    try {
      std::filesystem::create_directory(params.outputDirectory());
    } catch (std::exception& e) {
      Log::debug("Could not create output directory: {}", e.what());
      exit(EXIT_FAILURE);
    }
  }

  // set up scratch directory in tmp
  std::filesystem::path tmp_dir = details::outputSetup(seed);

  // setup the HDF5 writer instance
  details::writerSetup(params);
  HDF5Writer::getInstance().writeParams(params);

#pragma omp parallel for num_threads(params.numWorkers()) schedule(dynamic, 1)
  for (std::size_t rep_id = 0; rep_id < params.numReplicates(); ++rep_id) {
    // number of threads used by each worker
    omp_set_num_threads(params.numThreadsPerWorker());

    Params rep_params = params;
    rep_params.outputDirectory() = details::replicateSetup(tmp_dir, rep_id);
    if (params.shareInitState()) {
      rep_params.randomSeed() = seed;
      rep_params.postInitSeed() = details::shuffleSeed(seed, rep_id);
    } else {
      rep_params.randomSeed() = details::shuffleSeed(seed, rep_id);
    }
    rep_params.replicateId() = rep_id;

    runReplicate(rep_params);
  }

  if (params.deleteTemporaries()) std::filesystem::remove_all(tmp_dir);

  HDF5Writer::shutdown();
}

}  // namespace knoop
