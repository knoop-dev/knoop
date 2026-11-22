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

/**
 * \brief Parameters for the genome
 */
struct GenomeParams {
 private:
  std::size_t n_loci_;
  Eigen::VectorXd locus_freq_;
  Eigen::VectorXd locus_rec_;
  Eigen::VectorXd locus_mut_;

 public:
  GenomeParams(
      std::size_t n_loci,
      Eigen::VectorXd locus_freq,
      Eigen::VectorXd locus_rec,
      Eigen::VectorXd locus_mut)
      : n_loci_(n_loci),
        locus_freq_(std::move(locus_freq)),
        locus_rec_(std::move(locus_rec)),
        locus_mut_(std::move(locus_mut)) {
    if (locus_freq_.size() != n_loci_)
      throw KnoopDimensionError(
          "Locus allele frequency vector must be of length {} (got {})",
          n_loci_,
          locus_freq_.size());

    if (locus_rec_.size() != n_loci_)
      throw KnoopDimensionError(
          "Locus recombination probability vector must be of length {} "
          "(got {})",
          n_loci_,
          locus_rec_.size());

    if (locus_mut_.size() != n_loci_)
      throw KnoopDimensionError(
          "Locus mutation probability vector must be of length {} "
          "(got {})",
          n_loci_,
          locus_mut_.size());

    double prob_min = std::pow(2, -16);

    if (!checkVectorInRange(locus_freq_, prob_min, 1))
      throw KnoopInvalidArgumentError(
          "Locus allele frequencies must lie in [2^{{-16}}, 1]");

    if (!checkVectorInRange(locus_rec_, prob_min, 1))
      throw KnoopInvalidArgumentError(
          "Locus recombination probabilities must lie in [2^{{-16}}, 1]");

    if (!checkVectorInRange(locus_mut_, 0, 1))
      throw KnoopInvalidArgumentError(
          "Locus mutation probabilities must lie in [0, 1]");
  }

  /** Immutable accessors */

  std::size_t numLoci() const { return n_loci_; }

  const Eigen::VectorXd& locusFrequencies() const { return locus_freq_; }

  const Eigen::VectorXd& locusRecombination() const { return locus_rec_; }

  const Eigen::VectorXd& locusMutation() const { return locus_mut_; }

  /** Mutable accessors */

  std::size_t& numLoci() { return n_loci_; }

  Eigen::VectorXd& locusFrequencies() { return locus_freq_; }

  Eigen::VectorXd& locusRecombination() { return locus_rec_; }

  Eigen::VectorXd& locusMutation() { return locus_mut_; }
};

/**
 * \brief A single phenotype specification
 */
class PhenotypeParams;

class Phenotype {
  /** More concise PhenomeParams declaration... */
  friend PhenotypeParams;

 private:
  /** Name of the trait */
  std::string name_;

  /** Number of causal loci for this phenotype */
  std::size_t n_causal_loci_;

  /** Effect sizes of causal loci for this phenotype */
  Eigen::VectorXd effects_;

  /** Indices of causal loci for this phenotype */
  std::vector<std::size_t> causal_loci_;

  /** Variance of genetic component */
  double var_genetic_;

  /** Variance of environmental component */
  double var_environmental_;

  /** Variance of indirect component */
  double var_indirect_;

  /** \brief Fraction of indirect component attributable to paternal trait */
  double prop_indirect_paternal_;

 public:
  /**
   * \brief Create a new Phenotype instance.
   *
   * \param name Name used to represent phenotype.
   * \param n_causal_loci The number of causal loci for the phenotype.
   * \param effects The vector of effect sizes for the phenotype.
   * \param frep_dependence Frequency dependence of effect sizes. If specified,
   * effect sizes are scaled proportionately to the variance at the locus raised
   * to the power of this constant. If this argument is specified, the vector of
   * allele frequencies is also required as input.
   * \param allele_freq Vector of allele frequencies at loci. Required if the
   * option freq_dependence is specified, otherwise ignored.
   * \param causal_loci Vector of indices of causal loci. This is optional, but
   * can be used to manually specify which loci should be trait-associated for
   * this phenotype.
   * \param var_genetic Variance of the (additive) genetic component.
   * \param var_environmental Variance of the environmental component.
   * \param var_indirect Variance of the indirect (vertical) component.
   * \param prop_indirect_paternal Proportion of indirect component variance
   * that is attributable to the paternal trait.
   */
  Phenotype(
      std::string name,
      std::size_t n_causal_loci,
      const Eigen::VectorXd& effects,
      double freq_dependence = 0.0,
      const std::optional<Eigen::VectorXd>& allele_freq = std::nullopt,
      const std::optional<std::vector<std::size_t>>& causal_loci = std::nullopt,
      double var_genetic = 0.5,
      double var_environmental = 0.5,
      double var_indirect = 0.0,
      double prop_indirect_paternal = 0.5)
      : name_(std::move(name)),
        n_causal_loci_(n_causal_loci),
        effects_([effects, freq_dependence, allele_freq, var_genetic]() {
          Eigen::VectorXd eff = effects;

          if (freq_dependence != 0.0) {
            if (!allele_freq.has_value())
              throw KnoopInvalidArgumentError(
                  "Specifying frequency dependence during phenotype "
                  "declaration requires that the vector of locus allele "
                  "frequencies is supplied to Phenotype constructor");

            Eigen::ArrayXd weights = (2 * allele_freq.value().array() *
                                      (1 - allele_freq.value().array()))
                                         .pow(freq_dependence);

            eff.array() *= weights;
          }

          eff.normalize();
          eff *= std::sqrt(var_genetic);
          return eff;
        }()),
        causal_loci_(
            causal_loci.has_value() ? causal_loci.value()
                                    : std::vector<std::size_t>(n_causal_loci_)),
        var_genetic_(var_genetic),
        var_environmental_(var_environmental),
        var_indirect_(var_indirect),
        prop_indirect_paternal_(prop_indirect_paternal) {
    if (var_genetic_ < 0 || var_environmental_ < 0 || var_indirect_ < 0)
      throw KnoopRangeError(
          "Phenotype '{}': component variances must be non-negative", name_);

    if (prop_indirect_paternal_ < 0 || prop_indirect_paternal_ > 1)
      throw KnoopRangeError(
          "Trait paternal indirect proportion must lie in [0, 1]");

    if (effects_.size() != n_causal_loci_)
      throw KnoopInvalidArgumentError(
          "Effect sizes must number the amount of causal loci (expected: {}, "
          "got: {})",
          n_causal_loci_,
          effects_.size());
  }

  /** Immutable accessors */

  const std::string& name() const { return name_; }

  std::size_t numCausalLoci() const { return n_causal_loci_; }

  const Eigen::VectorXd& effects() const { return effects_; }

  const std::vector<std::size_t>& causalLoci() const { return causal_loci_; }

  double varGenetic() const { return var_genetic_; }

  double varEnvironmental() const { return var_environmental_; }

  double varIndirect() const { return var_indirect_; }

  double propIndirectPaternal() const { return prop_indirect_paternal_; }

  /** Mutable accessors */

  std::string& name() { return name_; }

  std::size_t& numCausalLoci() { return n_causal_loci_; }

  Eigen::VectorXd& effects() { return effects_; }

  std::vector<std::size_t>& causalLoci() { return causal_loci_; }

  double& varGenetic() { return var_genetic_; }

  double& varEnvironmental() { return var_environmental_; }

  double& varIndirect() { return var_indirect_; }

  double& propIndirectPaternal() { return prop_indirect_paternal_; }
};

class PhenotypeParams {
 private:
  std::vector<Phenotype> phenotypes_;
  std::vector<std::string> pheno_names_;
  std::unordered_map<std::string, std::size_t> pheno_idx_;
  Eigen::MatrixXd gen_cor_;
  Eigen::MatrixXd env_cor_;

 public:
  PhenotypeParams(
      std::vector<Phenotype> phenotypes,
      Eigen::MatrixXd gen_cor,
      Eigen::MatrixXd env_cor)
      : phenotypes_(std::move(phenotypes)),
        pheno_names_(phenotypes_.size()),
        pheno_idx_(phenotypes_.size()),
        gen_cor_(std::move(gen_cor)),
        env_cor_(std::move(env_cor)) {
    if (pheno_names_.size() != phenotypes_.size())
      throw KnoopGenericError(
          "Too few phenotype names (got {} against {} phenotypes)",
          pheno_names_.size(),
          phenotypes_.size());

    if (gen_cor_.rows() != phenotypes_.size() ||
        gen_cor_.cols() != phenotypes_.size())
      throw KnoopDimensionError(
          "Genetic component correlation is of incorrect dimension; expected "
          "{}x{} (got {}x{})",
          gen_cor_.rows(),
          gen_cor_.cols(),
          phenotypes_.size(),
          phenotypes_.size());

    if (env_cor_.rows() != phenotypes_.size() ||
        env_cor_.cols() != phenotypes_.size())
      throw KnoopDimensionError(
          "Environmental component correlation is of incorrect dimension; "
          "expected "
          "{}x{} (got {}x{})",
          env_cor_.rows(),
          env_cor_.cols(),
          phenotypes_.size(),
          phenotypes_.size());

    for (std::size_t pheno = 0; pheno < phenotypes_.size(); ++pheno) {
      pheno_names_[pheno] = phenotypes_[pheno].name_;
      pheno_idx_[phenotypes_[pheno].name_] = pheno;
    }
  }

  std::size_t size() const { return phenotypes_.size(); }

  std::size_t id(const std::string& name) const {
    if (!pheno_idx_.contains(name))
      throw KnoopInvalidArgumentError("Phenotype {} not found", name);
    return pheno_idx_.at(name);
  }

  const std::vector<std::string>& names() const { return pheno_names_; }

  const std::string& name(std::size_t id) const {
    return phenotypes_[id].name_;
  }

  const std::vector<Phenotype>& array() const { return phenotypes_; }

  Eigen::VectorXd varGenetic() const {
    Eigen::VectorXd result(size());
    for (std::size_t pheno = 0; pheno < size(); ++pheno)
      result(pheno) = varGenetic(pheno);

    return result;
  }

  Eigen::VectorXd varEnvironmental() const {
    Eigen::VectorXd result(size());
    for (std::size_t pheno = 0; pheno < size(); ++pheno)
      result(pheno) = varEnvironmental(pheno);

    return result;
  }

  Eigen::VectorXd varIndirect() const {
    Eigen::VectorXd result(size());
    for (std::size_t pheno = 0; pheno < size(); ++pheno)
      result(pheno) = varIndirect(pheno);

    return result;
  }

  std::vector<Eigen::VectorXd> effects() const {
    std::vector<Eigen::VectorXd> result(size());
    for (std::size_t pheno = 0; pheno < size(); ++pheno)
      result[pheno] = effects(pheno);

    return result;
  }

  std::vector<std::vector<std::size_t>> causalLoci() const {
    std::vector<std::vector<std::size_t>> result(size());
    for (std::size_t pheno = 0; pheno < size(); ++pheno)
      result[pheno] = causalLoci(pheno);

    return result;
  }

  std::vector<std::size_t> numCausalLoci() const {
    std::vector<std::size_t> result(size());
    for (std::size_t pheno = 0; pheno < size(); ++pheno)
      result[pheno] = numCausalLoci(pheno);

    return result;
  }

  double varGenetic(std::size_t id) const {
    return phenotypes_[id].var_genetic_;
  }

  double varEnvironmental(std::size_t id) const {
    return phenotypes_[id].var_environmental_;
  }

  double varIndirect(std::size_t id) const {
    return phenotypes_[id].var_indirect_;
  }

  double propIndirectPaternal(std::size_t id) const {
    return phenotypes_[id].prop_indirect_paternal_;
  }

  /**
   * These are potentially mutable-access fields, meaning they might be consumed
   * by simulation preprocessors. So have both const and non-const versions. Not
   * currently required for the other fields.
   */
  const Eigen::VectorXd& effects(std::size_t id) const {
    return phenotypes_[id].effects_;
  }

  Eigen::VectorXd& effects(std::size_t id) { return phenotypes_[id].effects_; }

  const std::vector<std::size_t>& causalLoci(std::size_t id) const {
    return phenotypes_[id].causal_loci_;
  }

  std::vector<std::size_t>& causalLoci(std::size_t id) {
    return phenotypes_[id].causal_loci_;
  }

  std::size_t numCausalLoci(std::size_t id) const {
    /** dangerous to pre-set – initial number might change */
    return phenotypes_[id].causal_loci_.size();
  }

  double varGenetic(const std::string& name) const {
    return varGenetic(id(name));
  }

  double varEnvironmental(const std::string& name) const {
    return varEnvironmental(id(name));
  }

  double varIndirect(const std::string& name) const {
    return varIndirect(id(name));
  }

  const Eigen::VectorXd& effects(const std::string& name) const {
    return effects(id(name));
  }

  Eigen::VectorXd& effects(const std::string& name) {
    return effects(id(name));
  }

  const std::vector<std::size_t>& causalLoci(const std::string& name) const {
    return causalLoci(id(name));
  }

  std::vector<std::size_t>& causalLoci(const std::string& name) {
    return causalLoci(id(name));
  }

  std::size_t numCausalLoci(const std::string& name) const {
    return numCausalLoci(id(name));
  }

  double propIndirectPaternal(const std::string& name) const {
    return propIndirectPaternal(id(name));
  }

  const Eigen::MatrixXd& geneticComponentCorrelation() const {
    return gen_cor_;
  }

  const Eigen::MatrixXd& environComponentCorrelation() const {
    return env_cor_;
  }
};

enum class MatingType { Random, Assortative };

class MatingParams {
 private:
  MatingType type_;
  Eigen::MatrixXd cor_;
  double tol_;
  std::size_t max_itr_;
  double temp_init_;
  double temp_decay_;
  bool exact_;
  bool whiten_;

 public:
  MatingParams(
      MatingType type = MatingType::Random,
      Eigen::MatrixXd cor = Eigen::MatrixXd{},
      double tol = 0.1,
      std::size_t max_itr = 0,
      double temp_init = 1.0,
      double temp_decay = 0.99,
      bool exact = false,
      bool whiten = true)
      : type_(type),
        cor_(std::move(cor)),
        tol_(tol),
        max_itr_(max_itr),
        temp_init_(temp_init),
        temp_decay_(temp_decay),
        exact_(exact),
        whiten_(whiten) {
    if (cor_.rows() != cor_.cols())
      throw KnoopDimensionError(
          "Spousal correlation matrix must be square (got {}x{})",
          cor_.rows(),
          cor_.cols());

    if (temp_init_ < 0)
      throw KnoopRangeError("Initial temperature must be positive");

    if (temp_decay_ < 0 || temp_decay > 1)
      throw KnoopRangeError("Temperature decay must lie in range (0, 1)");
  }

  /** Immutable accessors */

  const MatingType& type() const { return type_; }

  const Eigen::MatrixXd& spouseCorrelation() const { return cor_; }

  double tolerance() const { return tol_; }

  std::size_t maxIterations() const { return max_itr_; }

  double temperatureInit() const { return temp_init_; }

  double temperatureDecay() const { return temp_decay_; }

  bool exactObjective() const { return exact_; }

  bool whiten() const { return whiten_; }

  /** Mutable accessors */

  MatingType type() { return type_; }

  Eigen::MatrixXd& spouseCorrelation() { return cor_; }

  double& tolerance() { return tol_; }

  std::size_t& maxIterations() { return max_itr_; }

  double& temperatureInit() { return temp_init_; }

  double& temperatureDecay() { return temp_decay_; }

  bool& exactObjective() { return exact_; }

  bool& whiten() { return whiten_; }
};

class EstimatorParams {
 private:
  std::vector<PopulationEstimator> population_estimators_;
  std::vector<Sample> samples_;

 public:
  EstimatorParams(
      std::vector<PopulationEstimator> population_estimators,
      std::vector<Sample> samples)
      : population_estimators_(std::move(population_estimators)),
        samples_(std::move(samples)) {}

  /** Immutable accessors */

  const std::vector<PopulationEstimator>& estimators() const {
    return population_estimators_;
  }

  const std::vector<Sample>& samples() const { return samples_; }

  /** Mutable accessors */

  std::vector<PopulationEstimator>& estimators() {
    return population_estimators_;
  }

  std::vector<Sample>& samples() { return samples_; }
};

struct GlobalParams {
 private:
  std::size_t n_individuals_;
  std::size_t n_generations_;
  std::size_t n_replicates_;
  std::size_t n_workers_;
  std::size_t n_threads_per_worker_;
  bool share_init_state_;
  bool pedigree_warmup_;
  std::size_t pedigree_max_depth_;
  std::uint64_t random_seed_;
  std::filesystem::path output_dir_;
  std::optional<std::string> output_name_;
  LogLevel log_level_;
  bool log_to_file_;
  bool delete_temporaries_;

  std::size_t rep_id_;
  std::optional<std::uint64_t> post_init_seed_;

 public:
  GlobalParams(
      std::size_t n_individuals,
      std::size_t n_generations,
      std::size_t n_replicates,
      std::size_t n_workers,
      std::size_t n_threads_per_worker,
      bool share_init_state,
      bool pedigree_warmup,
      std::size_t pedigree_max_depth,
      std::uint64_t random_seed,
      std::filesystem::path output_dir,
      std::optional<std::string> output_name,
      LogLevel log_level = LogLevel::Info,
      bool log_to_file = false,
      bool delete_temporaries = true)
      : n_individuals_(n_individuals),
        n_generations_(n_generations),
        n_replicates_(n_replicates),
        n_workers_(n_workers),
        n_threads_per_worker_(n_threads_per_worker),
        share_init_state_(share_init_state),
        pedigree_warmup_(pedigree_warmup),
        pedigree_max_depth_(pedigree_max_depth),
        random_seed_(random_seed),
        output_dir_(std::move(output_dir)),
        output_name_(std::move(output_name)),
        log_level_(log_level),
        log_to_file_(log_to_file),
        delete_temporaries_(delete_temporaries) {}

  /** Immutable accessors */

  std::size_t numIndividuals() const { return n_individuals_; }

  std::size_t numGenerations() const { return n_generations_; }

  std::size_t numReplicates() const { return n_replicates_; }

  std::size_t numWorkers() const { return n_workers_; }

  std::size_t numThreadsPerWorker() const { return n_threads_per_worker_; }

  bool shareInitState() const { return share_init_state_; }

  bool pedigreeWarmup() const { return pedigree_warmup_; }

  std::size_t pedigreeMaxDepth() const { return pedigree_max_depth_; }

  std::uint64_t randomSeed() const { return random_seed_; }

  const std::filesystem::path& outputDirectory() const { return output_dir_; }

  const std::optional<std::string>& outputName() const { return output_name_; }

  LogLevel logLevel() const { return log_level_; }

  bool logToFile() const { return log_to_file_; }

  bool deleteTemporaries() const { return delete_temporaries_; }

  std::size_t replicateId() const { return rep_id_; }

  std::optional<std::uint64_t> postInitSeed() const { return post_init_seed_; }

  /** Mutable accessors */

  std::size_t& numIndividuals() { return n_individuals_; }

  std::size_t& numGenerations() { return n_generations_; }

  bool& pedigreeWarmup() { return pedigree_warmup_; }

  std::size_t& pedigreeMaxDepth() { return pedigree_max_depth_; }

  std::uint64_t& randomSeed() { return random_seed_; }

  std::filesystem::path& outputDirectory() { return output_dir_; }

  LogLevel& logLevel() { return log_level_; }

  bool& logToFile() { return log_to_file_; }

  std::size_t& replicateId() { return rep_id_; }

  std::optional<std::uint64_t>& postInitSeed() { return post_init_seed_; }
};

class Params {
 private:
  std::size_t n_individuals_;
  std::size_t n_generations_;
  std::size_t n_replicates_;
  std::size_t n_workers_;
  std::size_t n_threads_per_worker_;
  bool share_init_state_;
  bool pedigree_warmup_;
  std::size_t pedigree_max_depth_;
  std::uint64_t random_seed_;
  std::filesystem::path output_dir_;
  std::optional<std::string> output_name_;
  LogLevel log_level_;
  bool log_to_file_;
  bool delete_temporaries_;

  std::size_t rep_id_;
  std::optional<std::uint64_t> post_init_seed_;

  std::vector<PopulationEstimator> estimators_;
  std::vector<Sample> samples_;

 public:
  Params(
      std::size_t n_individuals,
      std::size_t n_generations,
      GenomeParams genome,
      PhenotypeParams phenotypes,
      MatingParams mating,
      std::vector<PopulationEstimator> estimators,
      std::vector<Sample> samples,
      std::size_t n_replicates,
      std::size_t n_workers,
      std::size_t n_threads_per_worker,
      bool share_init_state,
      bool pedigree_warmup,
      std::size_t pedigree_max_depth,
      std::uint64_t random_seed,
      std::filesystem::path output_dir,
      std::optional<std::string> output_name,
      LogLevel log_level = LogLevel::Info,
      bool log_to_file = false,
      bool delete_temporaries = true)
      : n_individuals_(n_individuals),
        n_generations_(n_generations),
        geno(std::move(genome)),
        pheno(std::move(phenotypes)),
        mate(std::move(mating)),
        estimators_(std::move(estimators)),
        samples_(std::move(samples)),
        n_replicates_(n_replicates),
        n_workers_(n_workers),
        n_threads_per_worker_(n_threads_per_worker),
        share_init_state_(share_init_state),
        pedigree_warmup_(pedigree_warmup),
        pedigree_max_depth_(pedigree_max_depth),
        random_seed_(random_seed),
        output_dir_(std::move(output_dir)),
        output_name_(std::move(output_name)),
        log_level_(log_level),
        log_to_file_(log_to_file),
        delete_temporaries_(delete_temporaries) {
    if (n_individuals_ == 0)
      throw KnoopInvalidArgumentError(
          "Population size (number of individuals) must be non-zero.");

    if (n_individuals_ % 2 != 0) n_individuals_ += 1;

    if (n_generations_ == 0)
      throw KnoopInvalidArgumentError(
          "Number of generations simulated must be non-zero.");

    for (const Phenotype& phenotype : pheno.array()) {
      if (phenotype.numCausalLoci() > geno.numLoci())
        throw KnoopRangeError(
            "Numer of causal loci for phenotype '{}' exceeds total amount of "
            "loci ({}: {}, total: {}); either lower the number of causal loci "
            "or increase the total number",
            phenotype.name(),
            phenotype.name(),
            phenotype.numCausalLoci(),
            geno.numLoci());
    }
  }

  GenomeParams geno;
  PhenotypeParams pheno;
  MatingParams mate;

  /** Immutable accessors */

  std::size_t numIndividuals() const { return n_individuals_; }

  std::size_t numGenerations() const { return n_generations_; }

  const std::vector<PopulationEstimator>& estimators() const {
    return estimators_;
  }

  const std::vector<Sample>& samples() const { return samples_; }

  std::size_t numReplicates() const { return n_replicates_; }

  std::size_t numWorkers() const { return n_workers_; }

  std::size_t numThreadsPerWorker() const { return n_threads_per_worker_; }

  bool shareInitState() const { return share_init_state_; }

  bool pedigreeWarmup() const { return pedigree_warmup_; }

  std::size_t pedigreeMaxDepth() const { return pedigree_max_depth_; }

  std::uint64_t randomSeed() const { return random_seed_; }

  const std::filesystem::path& outputDirectory() const { return output_dir_; }

  const std::optional<std::string>& outputName() const { return output_name_; }

  LogLevel logLevel() const { return log_level_; }

  bool logToFile() const { return log_to_file_; }

  bool deleteTemporaries() const { return delete_temporaries_; }

  std::size_t replicateId() const { return rep_id_; }

  std::optional<std::uint64_t> postInitSeed() const { return post_init_seed_; }

  /** Mutable accessors */

  std::size_t& numIndividuals() { return n_individuals_; }

  std::size_t& numGenerations() { return n_generations_; }

  std::vector<PopulationEstimator>& estimators() { return estimators_; }

  std::vector<Sample>& samples() { return samples_; }

  bool& pedigreeWarmup() { return pedigree_warmup_; }

  std::size_t& pedigreeMaxDepth() { return pedigree_max_depth_; }

  std::uint64_t& randomSeed() { return random_seed_; }

  std::filesystem::path& outputDirectory() { return output_dir_; }

  LogLevel& logLevel() { return log_level_; }

  bool& logToFile() { return log_to_file_; }

  std::size_t& replicateId() { return rep_id_; }

  std::optional<std::uint64_t>& postInitSeed() { return post_init_seed_; }
};

}  // namespace knoop
