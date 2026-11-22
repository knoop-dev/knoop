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

#include <benchmark/benchmark.h>
#include <knoop/core.h>
#include <knoop/estimate.h>
#include <knoop/init.h>
#include <knoop/mutate.h>
#include <knoop/sample.h>
#include <knoop/simulation.h>
#include <omp.h>
#include <sys/resource.h>

namespace details {

std::vector<double> generateSingularValues(
    std::size_t n_pheno,
    std::size_t asm_rank,
    double asm_decay,
    double asm_volume = 0.6) {
  double sval_0 =
      (asm_decay < 1.0)
          ? asm_volume * (1 - asm_decay) / (1 - std::pow(asm_decay, asm_rank))
          : asm_volume / static_cast<double>(asm_rank);

  std::vector<double> svals(n_pheno, 0.0);
  svals[0] = sval_0;

  for (std::size_t el = 1; el < asm_rank; ++el)
    svals[el] = asm_decay * svals[el - 1];

  return svals;
}

};  // namespace details

// NOLINTBEGIN

static void BM_GenerateHaplotypes_operator(benchmark::State& gbench_state) {
  knoop::random::Uniform dist_allele_freq(0.01, 0.5);

  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_loci = gbench_state.range(1);

  knoop::GenomeParams genome(
      n_loci,
      dist_allele_freq(n_loci),
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.0));

  knoop::PhenotypeParams phenotypes(
      {}, Eigen::MatrixXd::Identity(0, 0), Eigen::MatrixXd::Identity(0, 0));

  knoop::MatingParams mating(
      knoop::MatingType::Random, Eigen::MatrixXd::Identity(0, 0), 0, 0);

  knoop::Params params(
      n_individuals,
      1,  // n_generations
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},     // estimators
      {},     // samples
      1,      // n_replicates
      1,      // n_workers
      1,      // n_threads_per_worker
      false,  // share_init_state
      false,  // pedigree_warmup
      1,      // pedigree_max_depth
      42,     // random_seed
      std::filesystem::temp_directory_path(),
      std::nullopt);

  knoop::State state = knoop::buildState(params);
  knoop::GenerateHaplotypes haplo(params);

  for (auto _ : gbench_state) haplo(state);

  gbench_state.counters["genome_bytes"] = static_cast<double>(
      2 * state.geno().rows() * state.geno().words() * sizeof(std::uint64_t));
}

static void BM_GenotypeBuffer_computeAlleleFreqs(
    benchmark::State& gbench_state) {
  knoop::random::Uniform dist_allele_freq(0.01, 0.5);

  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_loci = gbench_state.range(1);

  knoop::GenomeParams genome(
      n_loci,
      dist_allele_freq(n_loci),
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.0));

  knoop::PhenotypeParams phenotypes(
      {}, Eigen::MatrixXd::Identity(0, 0), Eigen::MatrixXd::Identity(0, 0));

  knoop::MatingParams mating(
      knoop::MatingType::Random, Eigen::MatrixXd::Identity(0, 0), 0, 0);

  knoop::Params params(
      n_individuals,
      1,
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},
      {},
      1,
      1,
      1,
      false,
      false,
      1,
      42,
      std::filesystem::temp_directory_path(),
      std::nullopt);

  knoop::State state = knoop::buildState(params);
  knoop::GenerateHaplotypes haplo(params);
  haplo(state);

  for (auto _ : gbench_state) state.geno().computeLocusStats();
}

static void BM_GenotypeBuffer_computeLocusStats(
    benchmark::State& gbench_state) {
  knoop::random::Uniform dist_allele_freq(0.01, 0.5);

  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_loci = gbench_state.range(1);

  knoop::GenomeParams genome(
      n_loci,
      dist_allele_freq(n_loci),
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.0));

  knoop::PhenotypeParams phenotypes(
      {}, Eigen::MatrixXd::Identity(0, 0), Eigen::MatrixXd::Identity(0, 0));

  knoop::MatingParams mating(
      knoop::MatingType::Random, Eigen::MatrixXd::Identity(0, 0), 0, 0);

  knoop::Params params(
      n_individuals,
      1,
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},
      {},
      1,
      1,
      1,
      false,
      false,
      1,
      42,
      std::filesystem::temp_directory_path(),
      std::nullopt);

  knoop::State state = knoop::buildState(params);
  knoop::GenerateHaplotypes haplo(params);
  haplo(state);

  for (auto _ : gbench_state) state.geno().computeLocusStats();
}

static void BM_GenotypeBuffer_transpose(benchmark::State& gbench_state) {
  knoop::random::Uniform dist_allele_freq(0.01, 0.5);

  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_loci = gbench_state.range(1);

  knoop::GenomeParams genome(
      n_loci,
      dist_allele_freq(n_loci),
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.0));

  knoop::PhenotypeParams phenotypes(
      {}, Eigen::MatrixXd::Identity(0, 0), Eigen::MatrixXd::Identity(0, 0));

  knoop::MatingParams mating(
      knoop::MatingType::Random, Eigen::MatrixXd::Identity(0, 0), 0, 0);

  knoop::Params params(
      n_individuals,
      1,
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},
      {},
      1,
      1,
      1,
      false,
      false,
      1,
      42,
      std::filesystem::temp_directory_path(),
      std::nullopt);

  knoop::State state = knoop::buildState(params);
  knoop::GenerateHaplotypes haplo(params);
  haplo(state);

  for (auto _ : gbench_state) state.geno().transpose();
}

static void BM_UpdateGenome_operator(benchmark::State& gbench_state) {
  knoop::random::Uniform dist_allele_freq(0.01, 0.5);

  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_loci = gbench_state.range(1);

  knoop::GenomeParams genome(
      n_loci,
      dist_allele_freq(n_loci),
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.0));

  knoop::PhenotypeParams phenotypes(
      {}, Eigen::MatrixXd::Identity(0, 0), Eigen::MatrixXd::Identity(0, 0));

  knoop::MatingParams mating(
      knoop::MatingType::Random, Eigen::MatrixXd::Identity(0, 0), 0, 0);

  knoop::Params params(
      n_individuals,
      1,
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},
      {},
      1,
      1,
      1,
      false,
      false,
      1,
      42,
      std::filesystem::temp_directory_path(),
      std::nullopt);

  knoop::State state = knoop::buildState(params);
  knoop::GenerateHaplotypes haplo(params);
  knoop::Inherit inherit(params);

  haplo(state);
  state.transpose();

  std::iota(state.matching().begin(), state.matching().end(), 0);
  state.updatePedigree();

  for (auto _ : gbench_state) inherit(state);
}

static void BM_ScorePhenotypes_operator(benchmark::State& gbench_state) {
  knoop::random::Xoshiro256pp::setSeed(42);
  knoop::random::Uniform dist_allele_freq(0.01, 0.5);
  knoop::random::Normal dist_effects;

  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_loci = gbench_state.range(1);
  std::size_t n_phenotypes = gbench_state.range(2);

  std::vector<std::size_t> iota_loci(n_loci);
  std::iota(iota_loci.begin(), iota_loci.end(), 0);

  std::vector<knoop::Phenotype> phenotype_vec;
  phenotype_vec.reserve(n_phenotypes);
  for (std::size_t pheno = 0; pheno < n_phenotypes; ++pheno) {
    phenotype_vec.emplace_back(
        std::format("pheno{}", pheno),
        n_loci,
        dist_effects(n_loci),
        0.0,
        std::nullopt,
        iota_loci,
        0.5,
        0.5,
        0.0,
        0.5);
  }

  knoop::GenomeParams genome(
      n_loci,
      dist_allele_freq(n_loci),
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.0));

  knoop::PhenotypeParams phenotypes(
      std::move(phenotype_vec),
      Eigen::MatrixXd::Identity(n_phenotypes, n_phenotypes),
      Eigen::MatrixXd::Identity(n_phenotypes, n_phenotypes));

  knoop::MatingParams mating(
      knoop::MatingType::Random,
      Eigen::MatrixXd::Identity(n_phenotypes, n_phenotypes),
      0,
      0);

  knoop::Params params(
      n_individuals,
      1,
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},
      {},
      1,
      1,
      1,
      false,
      false,
      1,
      42,
      std::filesystem::temp_directory_path(),
      std::nullopt);

  knoop::State state = knoop::buildState(params);
  knoop::GenerateHaplotypes haplo(params);
  knoop::ScorePhenotypes score(params);

  haplo(state);
  state.geno().computeLocusStats();

  for (auto _ : gbench_state) {
    score(state);
    benchmark::DoNotOptimize(state.pheno()(knoop::Component::Total).data());
  }
}

static void BM_PhenotypeBuffer_computeStats(benchmark::State& gbench_state) {
  knoop::random::Uniform dist_allele_freq(0.01, 0.5);
  knoop::random::Normal dist_effects;

  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_loci = gbench_state.range(1);
  std::size_t n_phenotypes = gbench_state.range(2);

  std::vector<std::size_t> iota_loci(n_loci);
  std::iota(iota_loci.begin(), iota_loci.end(), 0);

  std::vector<knoop::Phenotype> phenotype_vec;
  phenotype_vec.reserve(n_phenotypes);
  for (std::size_t pheno = 0; pheno < n_phenotypes; ++pheno) {
    phenotype_vec.emplace_back(
        std::format("pheno{}", pheno),
        n_loci,
        dist_effects(n_loci),
        0.0,
        std::nullopt,
        iota_loci,
        0.5,
        0.5,
        0.0,
        0.5);
  }

  knoop::GenomeParams genome(
      n_loci,
      dist_allele_freq(n_loci),
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.0));

  knoop::PhenotypeParams phenotypes(
      std::move(phenotype_vec),
      Eigen::MatrixXd::Identity(n_phenotypes, n_phenotypes),
      Eigen::MatrixXd::Identity(n_phenotypes, n_phenotypes));

  knoop::MatingParams mating(
      knoop::MatingType::Random,
      Eigen::MatrixXd::Identity(n_phenotypes, n_phenotypes),
      0,
      0);

  knoop::Params params(
      n_individuals,
      1,
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},
      {},
      1,
      1,
      1,
      false,
      false,
      1,
      42,
      std::filesystem::temp_directory_path(),
      std::nullopt);

  knoop::State state = knoop::buildState(params);
  knoop::GenerateHaplotypes haplo(params);
  knoop::ScorePhenotypes score(params);

  haplo(state);
  state.geno().computeLocusStats();
  score(state);

  for (auto _ : gbench_state) {
    state.pheno().computeStats();
    benchmark::DoNotOptimize(state.pheno()(knoop::Component::Total).data());
  }
}

static void BM_RunReplicate(benchmark::State& gbench_state) {
  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_loci = gbench_state.range(1);
  std::size_t n_phenotypes = gbench_state.range(2);
  std::size_t n_generations = gbench_state.range(3);
  std::size_t n_threads = gbench_state.range(4);

  knoop::random::Normal dist_effects;

  std::vector<std::size_t> iota_loci(n_loci);
  std::iota(iota_loci.begin(), iota_loci.end(), 0);

  std::vector<knoop::Phenotype> phenotype_vec;
  phenotype_vec.reserve(n_phenotypes);
  for (std::size_t pheno = 0; pheno < n_phenotypes; ++pheno) {
    phenotype_vec.emplace_back(
        std::format("pheno{}", pheno),
        n_loci / n_phenotypes,
        dist_effects(n_loci / n_phenotypes),
        0.0,
        std::nullopt,
        std::nullopt,
        0.5,
        0.5,
        0.0,
        0.5);
  }

  knoop::GenomeParams genome(
      n_loci,
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.5),
      Eigen::VectorXd::Constant(n_loci, 0.0));

  knoop::PhenotypeParams phenotypes(
      std::move(phenotype_vec),
      Eigen::MatrixXd::Identity(n_phenotypes, n_phenotypes),
      Eigen::MatrixXd::Identity(n_phenotypes, n_phenotypes));

  knoop::MatingParams mating(
      knoop::MatingType::Assortative,
      knoop::utils::matrixFromSingularValues(
          details::generateSingularValues(n_phenotypes, n_phenotypes, 0.5)),
      0,
      std::max<std::size_t>(10 * n_individuals, 50000),
      1.0,
      0.99,
      true);

  knoop::Params params(
      n_individuals,
      n_generations,
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},
      {},
      1,  // n_replicates
      1,  // n_workers
      n_threads,
      false,
      false,
      1,
      42,
      std::filesystem::temp_directory_path() /
          std::format(
              "knoop_bench_replicate_{}",
              std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()),
      std::nullopt);

  for (auto _ : gbench_state) knoop::runSimulation(params);

  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);

  gbench_state.counters["n_individuals"] = n_individuals;
  gbench_state.counters["n_loci"] = n_loci;
  gbench_state.counters["n_phenotypes"] = n_phenotypes;
  gbench_state.counters["n_generations"] = n_generations;
  gbench_state.counters["n_threads"] = n_threads;

#ifdef __APPLE__
  gbench_state.counters["rss"] = static_cast<double>(ru.ru_maxrss);
#else
  gbench_state.counters["rss"] = static_cast<double>(ru.ru_maxrss) * 1024.0;
#endif
}

static void BM_AssortativeMating_operator(benchmark::State& gbench_state) {
  boost::random::xoshiro256pp& engine =
      knoop::random::Xoshiro256pp::getInstance();

  std::size_t n_individuals = gbench_state.range(0);
  std::size_t n_pheno = gbench_state.range(1);
  std::size_t asm_rank = gbench_state.range(2);
  double asm_decay = static_cast<double>(gbench_state.range(3)) / 100.0;

  std::vector<double> svals =
      details::generateSingularValues(n_pheno, asm_rank, asm_decay);

  std::size_t max_itr = std::max<std::size_t>(10 * n_individuals, 50000);

  double temp_decay = 0.99;

  // Create minimal phenotypes for mating benchmark
  std::vector<knoop::Phenotype> phenotype_vec;
  phenotype_vec.reserve(n_pheno);
  for (std::size_t pheno = 0; pheno < n_pheno; ++pheno) {
    phenotype_vec.emplace_back(
        std::format("pheno{}", pheno),
        1,
        Eigen::VectorXd::Constant(1, 1.0),
        0.0,
        std::nullopt,
        std::vector<std::size_t>{0},
        0.5,
        0.5,
        0.0,
        0.5);
  }

  knoop::GenomeParams genome(
      1,
      Eigen::VectorXd::Constant(1, 0.5),
      Eigen::VectorXd::Constant(1, 0.5),
      Eigen::VectorXd::Constant(1, 0.0));

  knoop::PhenotypeParams phenotypes(
      std::move(phenotype_vec),
      Eigen::MatrixXd::Identity(n_pheno, n_pheno),
      Eigen::MatrixXd::Identity(n_pheno, n_pheno));

  knoop::MatingParams mating(
      knoop::MatingType::Assortative,
      knoop::utils::matrixFromSingularValues(svals),
      0,        // tol
      max_itr,  // max_itr
      1.0,      // temp_init
      temp_decay,
      true);  // exact

  knoop::Params params(
      n_individuals,
      1,
      std::move(genome),
      std::move(phenotypes),
      std::move(mating),
      {},
      {},
      1,
      1,
      1,
      false,
      false,
      1,
      42,
      std::filesystem::temp_directory_path(),
      std::nullopt);

  knoop::State state = knoop::buildState(params);

  state.pheno().male() = Eigen::MatrixXd(n_individuals / 2, n_pheno);
  state.pheno().female() = Eigen::MatrixXd(n_individuals / 2, n_pheno);

  knoop::random::Normal::sample(
      engine, state.pheno().male().data(), (n_individuals / 2) * n_pheno);

  knoop::random::Normal::sample(
      engine, state.pheno().female().data(), (n_individuals / 2) * n_pheno);

  knoop::AssortativeMating mate(params);

  for (auto _ : gbench_state) {
    mate(state);
    benchmark::DoNotOptimize(state.pheno()(knoop::Component::Total).data());
  }

  gbench_state.counters["n_individuals"] = n_individuals;
  gbench_state.counters["n_pheno"] = n_pheno;
  gbench_state.counters["asm_rank"] = asm_rank;
  gbench_state.counters["asm_decay"] = asm_decay;
  gbench_state.counters["max_abs_error"] = mate.lInfError();
  gbench_state.counters["max_l2_error"] = mate.l2Error();
  gbench_state.counters["n_iterations"] =
      static_cast<double>(mate.numIterations());
}

#define knoop_BENCH_GENO_ARGS                 \
  Args({100, 50})                             \
      ->Args({1000, 500})                     \
      ->Args({5000, 2000})                    \
      ->Args({10000, 5000})                   \
      ->Args({50000, 5000})                   \
      ->Args({100000, 5000})                  \
      ->Args({1000000, 5000})                 \
      ->Args({10000000, 5000})                \
      ->Args({10000, 50000})                  \
      ->Args({10000, 100000})                 \
      ->Args({10000, 1000000})                \
      ->Args({10000, 10000000})               \
      ->ArgNames({"n_individuals", "n_loci"}) \
      ->Unit(benchmark::kMillisecond)

#define knoop_BENCH_PHENO_ARGS                           \
  Args({100, 50, 1})                                     \
      ->Args({100, 50, 5})                               \
      ->Args({100, 50, 20})                              \
      ->Args({1000, 500, 1})                             \
      ->Args({1000, 500, 5})                             \
      ->Args({1000, 500, 20})                            \
      ->Args({5000, 2000, 1})                            \
      ->Args({5000, 2000, 5})                            \
      ->Args({5000, 2000, 20})                           \
      ->Args({10000, 5000, 1})                           \
      ->Args({10000, 5000, 5})                           \
      ->Args({10000, 5000, 20})                          \
      ->Args({50000, 5000, 1})                           \
      ->Args({50000, 5000, 5})                           \
      ->Args({50000, 5000, 20})                          \
      ->Args({100000, 5000, 1})                          \
      ->Args({100000, 5000, 5})                          \
      ->Args({100000, 5000, 20})                         \
      ->Args({1000000, 5000, 1})                         \
      ->Args({1000000, 5000, 5})                         \
      ->Args({1000000, 5000, 20})                        \
      ->Args({10000, 50000, 1})                          \
      ->Args({10000, 50000, 5})                          \
      ->Args({10000, 50000, 20})                         \
      ->Args({10000, 100000, 1})                         \
      ->Args({10000, 100000, 5})                         \
      ->Args({10000, 100000, 20})                        \
      ->ArgNames({"n_individuals", "n_loci", "n_pheno"}) \
      ->Unit(benchmark::kMillisecond)

#define knoop_BENCH_MATING_ARGS                                            \
  ArgsProduct({{1000, 10000, 100000, 1000000, 10000000}, {5}, {5}, {100}}) \
      ->ArgsProduct(                                                       \
          {{1000, 10000, 100000, 1000000, 10000000}, {10}, {10}, {100}})   \
      ->ArgsProduct(                                                       \
          {{1000, 10000, 100000, 1000000, 10000000}, {20}, {20}, {100}})   \
      ->ArgNames({"n_individuals", "n_pheno", "asm_rank", "asm_decay"})    \
      ->Iterations(1)                                                      \
      ->Repetitions(5)                                                     \
      ->ReportAggregatesOnly()                                             \
      ->Unit(benchmark::kMillisecond)

BENCHMARK(BM_GenerateHaplotypes_operator)->knoop_BENCH_GENO_ARGS;
BENCHMARK(BM_GenotypeBuffer_computeAlleleFreqs)->knoop_BENCH_GENO_ARGS;
BENCHMARK(BM_GenotypeBuffer_computeLocusStats)->knoop_BENCH_GENO_ARGS;
BENCHMARK(BM_GenotypeBuffer_transpose)->knoop_BENCH_GENO_ARGS;
BENCHMARK(BM_UpdateGenome_operator)->knoop_BENCH_GENO_ARGS;
BENCHMARK(BM_ScorePhenotypes_operator)->knoop_BENCH_PHENO_ARGS;
BENCHMARK(BM_PhenotypeBuffer_computeStats)->knoop_BENCH_PHENO_ARGS;

BENCHMARK(BM_RunReplicate)
    // comparison cells: population sweep
    ->ArgsProduct(
        {{1000, 5000, 20000, 100000},
         {1000},
         {5, 10, 20},
         {5},
         {1, 2, 4, 8, 12}})
    // comparison cells: genome sweep
    ->ArgsProduct(
        {{10000},
         {1000, 2000, 5000, 10000},
         {5, 10, 20},
         {5},
         {1, 2, 4, 8, 12}})
    // knoop only: large populations
    ->ArgsProduct({{1000000, 10000000}, {2000}, {5}, {5}, {1, 2, 4, 8, 12}})
    ->ArgNames(
        {"n_individuals",
         "n_loci",
         "n_phenotypes",
         "n_generations",
         "n_threads"})
    ->Iterations(1)
    ->Name("BM_RunReplicate")
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_AssortativeMating_operator)->knoop_BENCH_MATING_ARGS;

BENCHMARK_MAIN();

// NOLINTEND
