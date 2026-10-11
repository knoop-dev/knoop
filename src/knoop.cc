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

#include <knoop/core.h>
#include <knoop/estimate.h>
#include <knoop/init.h>
#include <knoop/io.h>
#include <knoop/simulation.h>
#include <getopt.h>
#include <wordexp.h>

#include <Eigen/Dense>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

constexpr std::string getOSName() {
  std::string os = knoop_OS;
  if (os == "Darwin") return "macOS";
  return os;
}

enum Context : uint32_t {
  // Base domain
  Global = 0,
  Genome = 1 << 0,
  Phenotype = 1 << 1,
  Mating = 1 << 2,
  Sample = 1 << 3,
  SampleEstimator = 1 << 4,
  PopulationEstimator = 1 << 5,

  // Subcomponent
  InitAlleleFreqs = 1 << 6,
  RecombinationProbs = 1 << 7,
  MutationProbs = 1 << 8,
  CausalLoci = 1 << 9,
  GeneticComponent = 1 << 10,
  EnvironmentalComponent = 1 << 11,
  EffectSizes = 1 << 12,
  MatrixSpecification = 1 << 13,

  // Derived contexts
  GenomeInitAlleleFreqs = Genome | InitAlleleFreqs,
  GenomeRecombinationProbs = Genome | RecombinationProbs,
  GenomeMutationProbs = Genome | MutationProbs,
  GenomeProbabilities = InitAlleleFreqs | RecombinationProbs | MutationProbs,
  PhenotypeCausalLoci = Phenotype | CausalLoci,
  PhenotypeEffectSizes = Phenotype | EffectSizes,
  PhenotypeGeneticCorrelation =
      Phenotype | GeneticComponent | MatrixSpecification,
  PhenotypeEnvironmentalCorrelation =
      Phenotype | EnvironmentalComponent | MatrixSpecification,
  MatingCorrelation = Mating | MatrixSpecification,

  // Base domain
  ContextDomain = Global | Genome | Phenotype | Mating | Sample |
                  SampleEstimator | PopulationEstimator,
};

Context contextDomain(Context context) {
  return static_cast<Context>(context & ContextDomain);
}

void checkContext(Context context, Context expected, const std::string& flag) {
  if (contextDomain(context) != expected)
    throw std::runtime_error(
        std::format("Unexpected flag '{}' in current context", flag));
}

enum Option {
  GlobalHelp,
  GlobalVersion,
  GlobalNumIndividuals,
  GlobalNumGenerations,
  GlobalNumThreads,
  GlobalNumWorkers,
  GlobalNumThreadsPerWorker,
  GlobalNumReplicates,
  GlobalRandomSeed,
  GlobalOutputDirectory,
  GlobalOutputName,
  GlobalLogLevel,
  GlobalLogNoFile,
  GlobalPedigreeMaxDepth,
  GlobalPedigreeWarmup,
  GlobalSaveConfig,
  GlobalLoadConfig,
  GlobalShareInitState,
  GlobalNoDeleteTmp,
  GlobalNoRun,
  GenomeLocusInitFreqs,
  GenomeLocusRecombinationProbs,
  GenomeLocusMutationProbs,
  PhenotypeSpec,
  PhenotypeVarGenetic,
  PhenotypeVarEnvironmental,
  PhenotypeVarVertical,
  PhenotypeLocusEffects,
  PhenotypeLocusIndices,
  PhenotypeGeneticCor,
  PhenotypeEnvironmentalCor,
  MatingSpec,
  MatingCor,
  MatingExact,
  MatingErrorTolerance,
  MatingMaxIterations,
  MatingAnnealingTempInit,
  MatingAnnealingTempDecay,
  SampleSpec,
  SampleProbandType,
  SampleNumProbands,
  SampleWeightFunction,
  SampleWeightOnPhenotypes,
  SampleWeightOfMembers,
  SampleWeightAggregation,
  SampleIncludeMembers,
  SampleEstimators,
  SampleEstimatorSpec,
  SampleEstimatorType,
  SampleEstimatorExec,
  SampleEstimatorNumRows,
  SampleEstimatorNumCols,
  SampleEstimatorRowNames,
  SampleEstimatorColNames,
  PopulationEstimatorSpec,
  VirtualNumLoci,  // these options are context-sensitive
  VirtualFile,
  VirtualValue,
  VirtualDistribution,
  VirtualSingularValues
};

std::string phenotypeDocs() {
  return R"(
--phenotype <name> declares a new phenotype and opens its context. Every
flag below, up to the next --phenotype (or a flag from a different domain),
configures that phenotype. Flags --pheno-gen-cor and --pheno-env-cor assign
correlation structure for the genetic and environmental components of traits,
respectively.

  --phenotype <name>
    --n-loci <n_loc>
    --loci [--value <idx> | --file <path>]
    --effects [--value <val> | --file <path> | --dist <distribution>]
    --var-genetic <var_gen>
    --var-environmental <var_env>
    --var-vertical <var_vert>
  --pheno-gen-cor [--value <val(s)...> | --file <path> | --singular-values <sval(s)...>]
  --pheno-env-cor [--value <val(s)...> | --file <path> | --singular-values <sval(s)...>]

A phenotype is the sum of up to three independent components:

    phenotype = genetic + environmental + vertical

--var-genetic, --var-environmental and --var-vertical set the share of
variance contributed by each; they need not sum to 1, since they are
renormalised internally so that the resulting heritabilities sum to one.
The genetic component is a linear combination of allele dosages at <n_loc>
causal loci, weighted by an effect-size vector; that vector is rescaled
internally so the genetic component alone carries variance h2_genetic. The
environmental component is drawn from an iid normal distribution for each
individual.

  --loci
    Which of the genome's <n_loci> loci are causal for this phenotype, given
    as indices, either directly via --value, or loaded from a file via --file.
    If omitted, <n_loc> loci are chosen at random. Mutually exclusive with
    --pheno-gen-cor / --pheno-env-cor below.

  --effects
    Per-locus effect sizes for the genetic component, over the causal loci above.
    Either a constant (--value), a vector loaded from file (--file), or drawn from
    a distribution via --dist, e.g. normal(0,1). Defaults to a Rademacher
    distribution (+-1 with equal probability) if omitted.

With more than one phenotype declared, --pheno-gen-cor and --pheno-env-cor set
the cross-phenotype correlation of the genetic and environmental components
respectively -- e.g. two phenotypes sharing part of their genetic basis (pleiotropy)
without either being causal for the other's loci. Each is a P x P correlation matrix
over the P declared phenotypes, read symmetrically (component correlation, not
directional):

             height weight
    height [ 1.00   0.50 ]
    weight [ 0.50   1.00 ]

Supply it directly (--value / --file), or generate a random one with a prescribe
eigenspectrum via --singular-values (P values in, a random symmetric P x P matrix
with exactly those eigenvalues out). Defaults to the identity matrix (uncorrelated
components) if not specified.

For more on the --value / --file / --dist / --singular-values flags themselves,
run --help on each.
  )";
}

std::string matingDocs() {
  return R"(
--mating {random | assortative} selects how mates are paired each generation.

  random
    Mates are drawn uniformly at random from the opposite sex. No sub-flags
    apply. This is the default if --mating is never specified.

  assortative
    Mates are paired so that the phenotypic correlation among mated pairs
    matches a target correlation matrix, --mate-cor.

    --mate-cor [--value <vals> | --file <path> | --singular-values <sv>]
      The target correlation matrix among mates, over the P declared
      phenotypes. Entry (i, j) is the correlation between the husband's
      phenotype i and the wife's phenotype j; unlike --pheno-gen-cor /
      --pheno-env-cor, this matrix is not forced symmetric, so cross-trait
      and directional assortment are both expressible:

                          wife:
                          height weight
          husband height [ 0.40  0.10 ]
                  weight [ 0.05  0.20 ]

      A single phenotype reduces this to the familiar scalar spousal correlation.

    --tol-inf <tol_inf>
      Convergence tolerance on the empirical-vs-target mate correlation
      (matrix norm of the difference). Default 1e-7.

    --max-itr <max_itr>
      Maximum number of annealing proposals attempted before giving up and
      returning the best pairing found so far. Default 2,000,000.

    --temp-init <temp_init>
    --temp-decay <temp_decay>
      Initial temperature and per-iteration decay of the mate matching algorithm.
      A higher --temp-init (default: 1.0) accepts more correlation-worsening swaps
      early on, helping escape local optima at the cost of slower convergence;
      --temp-decay (default 0.99) controls how quickly that exploration cools into
      pure hill-climbing. Each iteration of the algorithm, the temperature is
      updated via temp *= temp_decay.

    --exact
      If enabled, matches spouse pairs so as to produce identical correlation
      structure to <mate-cor>. Otherwise, each generation a random matrix is
      drawn from a matrix normal distribution around <mate-cor>. This follows
      from the probabilistic generative model X_female = R^T * X_male + E, where
      R is the spousal correlation matrix <mate-cor>. This induces realistic
      variation attributable to the finite size of the population, whereas the
      exact case is more useful when the model assumes that the correlation
      matrix is assumed identically within each generation.
  )";
}

std::string sampleDocs() {
  return R"(
knoop models ascertainment / participation bias by drawing a sub-sample of
the population with inclusion probability governed by a weighting function of
(possibly aggregated) phenotype data. Two related declarations make this up:

  --sample <name>
    --proband {self | mate | family}
    --n-probands <n_prob>
    --on <phenotype(s)...>
    --of <member(s)...>
    --include <member(s)...>
    --agg {mean | max | min | identity}
    --weight {uniform() | logistic(<coef(s)...>, ...) |
              case-control(<thresh>, {<|>}, <coef(s)...>, ...)}
    --estimators <estimator(s)...>

  --sample-estimator <name>
    --type {gwas(<n_pcs>, <pval_thresh>) | haseman-elston | greml | external |
            sample-mean | sample-var | sample-cov | sample-mate-cor}
    --exec <exec_cmd>                                       (type external only)
    --n-rows <n_rows>                                       (type external only)
    --n-cols <n_cols>                                       (type external only)
    --row-names <name(s)...>                                (type external only)
    --col-names <name(s)...>                                (type external only)

--proband fixes the unit of sampling and, with it, which family members are
addressable through --of and --include:

  self     a single person (--of is not applicable)
  mate     a mated couple: husband, wife, and their in-laws. Valid options are
            {'husband', 'wife', 'husband-father', 'husband-mother',
            'husband-sister', 'husband-brother-in-law', 'wife-father',
            'wife-mother', 'wife-brother', 'wife-sister-in-law', 'couple',
            'husband-siblings', 'wife-siblings', 'husband-in-laws',
            'wife-in-laws', 'parents', 'husband-family', 'wife-family',
            'males', 'females', 'all'}.
  family   a nuclear family: father, mother, their children, and whoever their
            children married. Valid options are {'father', 'mother', 'son',
            'daughter', 'son-wife', 'daughter-husband', 'parents', 'siblings',
            'males', 'females', 'all'}.

For each proband, --on selects which phenotype(s) enter the weighting calculation
and --of selects whose values are used; --agg then reduces the values across the
selected members down to one number per phenotype per proband (mean/max/min, or
identity for a single member). --weight turns that aggregated vector into an
inclusion probability:

  uniform()             every proband is equally likely to be sampled -- i.e.,
                         no ascertainment bias, a random sample.
  logistic(b1, b2, ...) inclusion probability = 1 / (1 + exp(-agg * b)),
                         a logistic model in the aggregated phenotype
                         values; one coefficient per value selected via
                         --on, so ascertainment can depend on, and bias
                         estimates with respect to, one or more phenotypes.
  case-control(t, {<|>}, b1, b2, ...)
                         hard case/control cutoff: probands are included
                         iff (agg * b) is above t (>) or below t (<).

knoop then draws <n_probands> sample units under that probability, with the
members listed with --include being those who are written to file. Subsequently
attaches every sample estimator named in <estimator(s)...> (each declared
separately via its own --sample-estimator block) to the resulting sub-sample, so
the same estimator machinery used at the population level can be recomputed
under whatever ascertainment or sampling scheme is being studied.

--sample-estimator works like --estimator (see --help --estimator) but is declared
by name so it can be referenced from one or more --sample blocks via --estimators;
--type additionally supports sample-only estimators (sample-mean, sample-var,
sample-cov, sample-mate-cor) and an external escape hatch (--type external).

For type external, knoop writes the sample's genotypes as a standard PLINK1
binary fileset (<sample_dir>/data.bed / .bim / .fam / .pheno) and invokes --exec
as:

  <exec> --bfile <sample_dir>/data --out <sample_dir>/results_<name>

exactly the same convention the built-in plink2/gcta64-backed estimators
(gwas, greml, haseman-elston) use internally -- --exec just needs to be something
that reads a --bfile-style PLINK prefix. Whatever program you point --exec at
must write its result to the --out path as plain text in the same matrix format
--file accepts elsewhere (values comma- or space-separated, rows separated by a
semicolon or newline), sized --n-rows x --n-cols; knoop reads that file back in
once --exec has finished running.
  )";
}

void displayVersion() {
  std::string version_str = std::format(
      "knoop v{} {} {} ({} build, {})",
      knoop_VERSION,
      getOSName(),
      knoop_ARCH,
      knoop_BUILD_TYPE,
      __DATE__);

  std::cout << version_str << std::endl;
}

void displayHeader() {
  std::string header_str = std::format(
      "{:<40}{:>40}\n{:<40}{:>40}",
      std::format(
          "knoop v{} {} {} ({} build)",
          knoop_VERSION,
          getOSName(),
          knoop_ARCH,
          knoop_BUILD_TYPE),
      "https://github.com/lvthnn/knoop",
      "(C) 2025-2026 Kári Hlynsson",
      "GNU General Public License v3");

  std::cout << header_str << std::endl;
}

void displayHelp() {
  std::string help_str = R"(
For full documentation and usage examples, see https://github.com/lvthnn/knoop.
Running knoop --help <option(s)...> prints usage guides on the various command
flags listed below. Example: knoop --help --n-individuals.

This software is under development and may contain bugs. Please report any
issues you may experience on GitHub or by email at kari.hlynsson26@imperial.ac.uk.

program options:
  -h, --help
  -v, --version

global options:
  --n-individuals <n_ind>
  --n-loci <n_loc>
  --n-generations <n_gen>
  --n-replicates <n_rep>
  --n-workers <n_wrk>
  --n-threads-per-worker <n_thr>
  --random-seed <rng_seed>
  --output-dir <out_dir>
  --output-name <out_name>
  --pedigree-max-depth <max_depth>
  --pedigree-warmup
  --log-level <log_level>
  --log-to-stdout
  --save-config
  --load-config <config_path>
  --share-init-state
  --no-delete-tmp
  --no-run

genome options:
  --locus-freq [--value <val> | --file <path> | --dist <dist>]
  --locus-rec [--value <val> | --file <path> | --dist <dist>]
  --locus-mut [--value <val> | --file <path> | --dist <dist>]

phenome options:
  --phenotype <pheno_name>
    --var-genetic <var_gen>
    --var-environmental <var_env>
    --var-vertical <var_vert>
    --n-loci <n_loc>
    --loci <loc>
    --effects <eff>
  --pheno-gen-cor [--value <val(s)...> | --file <path> | --singular-values <sval(s)...>]
  --pheno-env-cor [--value <val(s)...> | --file <path> | --singular-values <sval(s)...>]

mating options:
  --mating {assortative | random}
    --mate-cor [--value <val(s)...> | --file <path> | --singular-values <sval(s)...>]
    --tol-inf <tol-inf>
    --max-itr <max-itr>
    --temp-init <tmp_init>
    --temp-decay <tmp_decay>
    --exact

sampling and estimation options:
  --estimator {genotype-mean() | genotype-var() | genotype-freq() |
               genotype-cov() | genotype-cor() | allele-freq() | heritability() |
               pheno-mean(<component>) | pheno-var(<component>) |
               pheno-cov(<component>) | pheno-cor(<component>) |
               mate-cor(<component>) | sibling-cov(<component>) |
               sibling-cor(<component>) | cousin-cov(<component>, <degree>) |
               cousin-cor(<component>, <degree>) |
               ancestor-cov(<component>, <degree>) |
               ancestor-cor(<component>, <degree>)}
  --sample-estimator <name>
    --type {gwas(<n_pcs>, <pval_thresh>) | haseman-elston | greml | external |
            sample-mean | sample-var | sample-cov | sample-mate-cor}
    --exec <exec_cmd>
    --n-rows <n_rows>
    --n-cols <n_cols>
    --row-names <row_name(s)...>
    --col-names <col_name(s)...>
  --sample <sample_name>
    --proband {self | mate | family}
    --n-probands <n-prob>
    --weight {logistic(<coef(s)...>) | uniform() |
              case-control(<thresh>, {<|>}, <coef(s)...>)}
    --on <phenotype(s)...>
    --of <member(s)...>
    --include <member(s)...>
    --agg <agg_fn>
    --estimators <est_names>
  )";

  std::cout << help_str << std::endl;
}

void displayHelpShort() {
  std::string help_str = R"(
  knoop <option(s)...>
  knoop --load-config <config_file> <option(s)...>
  knoop --help <option(s)...>

To see all options, run "knoop --help".)";

  std::cout << help_str << std::endl;
}

void displayHelpOption(Option option, const std::string& option_flag) {
  std::unordered_map<Option, std::string> docs;

  docs[Option::GlobalHelp] = R"(
:-)
  )";

  docs[Option::GlobalVersion] = R"(
Displays version and build information.
  )";

  docs[Option::GlobalNumIndividuals] = R"(
Number of individuals to simulate in each generation. Positive, non-zero integer.
  )";

  docs[Option::GlobalNumGenerations] = R"(
Number of generations to simulate. Positive, non-zero integer.
  )";

  docs[Option::GlobalNumWorkers] = R"(
Number of threads to use for multithreaded replicate simulation. Positive, non-zero
integer.
  )";

  docs[Option::GlobalNumThreadsPerWorker] = R"(
Number of threads used per replicate worker (see --n-workers). Positive, non-zero
integer.
  )";

  docs[Option::GlobalNumReplicates] = R"(
Number of replicate simulations to perform. Can be started from either the same
initial population or different ones based on whether the --share-init-state flag
is specified. Positive, non-zero integer.
  )";

  docs[Option::GlobalRandomSeed] = R"(
Random seed for reproducible results. Positive, non-zero integer.
  )";

  docs[Option::GlobalOutputDirectory] = R"(
Path to output directory of generated results.
  )";

  docs[Option::GlobalOutputName] = R"(
Basename used for files produced during simulation.
  )";

  docs[Option::GlobalLogLevel] = R"(
Specifies the verbosity of knoop's internal logger system. Options are 'none',
'error', 'warning', 'info', 'debug'.
  )";

  docs[Option::GlobalLogNoFile] = R"(
If enabled, knoop logs directly to the shell instead of writing to file.
  )";

  docs[Option::GlobalPedigreeMaxDepth] = R"(
Maximum depth of pedigree structure, used when working with kinship-based
estimators such as ancestor-cov and cousin-cov. For a former such estimator of
degree <degree>, --pedigree-max-depth must be configured to be at least
<degree>, whereas for the latter a minimum of <degree> + 1 is required.
  )";

  docs[Option::GlobalPedigreeWarmup] = R"(
If enabled, simulates panmictic generations before generating the founder sibling
pairs so that the pedigree is at full capacity immediately in the first generation.
This option is useful when employing kinship-based estimators so estimates are
available immediately from the initial generation. Otherwise, the estimators
return NA until the pedigree is filled to its desired depth.

NOTE: ancestor-cov estimators currently produce NA values for the first <degree>
generations irrespective of whether --pedigree-warmup is enabled. This is a known
issue and will be fixed soon.
  )";

  docs[Option::GlobalSaveConfig] = R"(
If enabled, saves configuration to a TOML file for reuse. Can be paired with the
--no-run flag to generate a configuration file without performing simulation.
  )";

  docs[Option::GlobalLoadConfig] = R"(
Attempts to load configuration file from the provided path. Parameters can be
overloaded by supplying them in conjunction with this option. In the case that a
declarative option is specified, such as adding a phenotype, then this is added
to the existing configuration.
  )";

  docs[Option::GlobalShareInitState] = R"(
If enabled, starts simulations from the same founder population and state.
  )";

  docs[Option::GlobalNoDeleteTmp] = R"(
If enabled, temporary files produced during simulation are not deleted once the
run has completed. To access the path of the scratch directory, run knoop with
--log-level debug and see the log.
)";

  docs[Option::GlobalNoRun] = R"(
If enabled, the software exits without performing simulation. This is useful when
specifying configurations to be saved to file via --save-config.
  )";

  docs[Option::GenomeLocusInitFreqs] = R"(
Specify the allele frequencies (AFs) of the genetic loci in the simulation.
In a population numbering <n_ind> individuals, the AF of a locus is the total
dosage of the alternate allele divided by the total number of alleles for the
locus, i.e., 2 * <n_ind>.

The frequencies can be specified through directly through the --value flag, in which
case a constant double value in the range [0.0, 1.0] must be specified. On the other
hand, if one wishes for the frequencies to vary by locus, these data may be supplied
through the --file option, which takes as argument a path to a file. Lastly, the
--dist flag can be used to specify a probability distribution from which these
frequencies are drawn.

For more information on virtual flags and their use, run --help on flags --value,
--file, --dist, and --singular-values.
  )";

  docs[Option::GenomeLocusRecombinationProbs] = R"(
Specify the recombination probabilities of the genetic loci in the simulation.
Each entry gives the probability that a gamete formed at meiosis recombines at
that position rather than continuing to inherit from the same parental strand. A
value of 0.5 corresponds to free recombination or independence between adjoining
loci, whereas a value near 0.0 corresponds to tight linkage.

As with --locus-freq, these can be specified directly through the --value flag
(<n_loci> values in [0.0, 1.0]), loaded from a file via --file, or drawn from a
distribution via --dist. Since these are probabilities, --dist is restricted to
[0,1]-supported distributions (uniform, beta).

For more information on virtual flags and their use, run --help on flags --value,
--file, and --dist.
  )";

  docs[Option::GenomeLocusMutationProbs] = R"(
Specify the per-locus mutation probabilities, i.e., the chance that a given allele
is flipped to its complement when passed from parent to offspring. One value per
locus, in [0.0, 1.0], with default 0.0 (no mutation).

As with --locus-freq and --locus-rec, these can be specified directly through the
--value flag, loaded from a file via --file, or drawn from a distribution via
--dist, restricted to [0,1]-supported distributions (uniform, beta) since these
are probabilities.

For more information on virtual flags and their use, run --help on flags --value,
--file, and --dist.
  )";

  docs[Option::PhenotypeSpec] = phenotypeDocs();
  docs[Option::PhenotypeVarGenetic] = phenotypeDocs();
  docs[Option::PhenotypeVarEnvironmental] = phenotypeDocs();
  docs[Option::PhenotypeVarVertical] = phenotypeDocs();
  docs[Option::PhenotypeLocusEffects] = phenotypeDocs();
  docs[Option::PhenotypeLocusIndices] = phenotypeDocs();
  docs[Option::PhenotypeGeneticCor] = phenotypeDocs();
  docs[Option::PhenotypeEnvironmentalCor] = phenotypeDocs();

  docs[Option::MatingSpec] = matingDocs();
  docs[Option::MatingCor] = matingDocs();
  docs[Option::MatingErrorTolerance] = matingDocs();
  docs[Option::MatingMaxIterations] = matingDocs();
  docs[Option::MatingAnnealingTempInit] = matingDocs();
  docs[Option::MatingAnnealingTempDecay] = matingDocs();

  docs[Option::SampleSpec] = sampleDocs();
  docs[Option::SampleProbandType] = sampleDocs();
  docs[Option::SampleNumProbands] = sampleDocs();
  docs[Option::SampleWeightFunction] = sampleDocs();
  docs[Option::SampleWeightOnPhenotypes] = sampleDocs();
  docs[Option::SampleWeightOfMembers] = sampleDocs();
  docs[Option::SampleWeightAggregation] = sampleDocs();

  docs[Option::SampleEstimators] = sampleDocs();
  docs[Option::SampleEstimatorSpec] = sampleDocs();
  docs[Option::SampleEstimatorType] = sampleDocs();
  docs[Option::SampleEstimatorNumRows] = sampleDocs();
  docs[Option::SampleEstimatorNumCols] = sampleDocs();
  docs[Option::SampleEstimatorRowNames] = sampleDocs();
  docs[Option::SampleEstimatorColNames] = sampleDocs();

  docs[Option::PopulationEstimatorSpec] = R"(
Declares a population-wide estimator, or a statistic computed once per generation
over the full population. Repeat the flag to add more than one.

  --estimator {genotype-mean() | genotype-var() | genotype-freq() |
               genotype-cov() | genotype-cor() | allele-freq() | heritability() |
               pheno-mean(<component>) | pheno-var(<component>) |
               pheno-cor(<component>) | pheno-cov(<component>) |
               mate-cor(<component>) | sibling-cov(<component>) |
               sibling-cor(<component>) | cousin-cov(<component>, <degree>) |
               cousin-cor(<component>, <degree>) |
               ancestor-cov(<component>, <degree>) |
               ancestor-cor(<component>, <degree>)}

genotype-* and heritability take no arguments and operate on the genome buffer /
true heritability directly. The rest operate on a phenotype component and take an
optional <component>, one of genetic, environmental, vertical, or total. The
default component selected is total.

cousin-cov and ancestor-cov additionally take an optional <degree> (default: 1):
the pedigree distance to compute the covariance at degree 1 for first cousins /
grandparent-grandchild, degree 2 for second cousins / great-grandparents, and so
on. Resolving degree <degree> requires a pedigree deep enough to reach it; see
--pedigree-max-depth and --pedigree-warmup. NOTE: As of the current version,
ancestor-cov estimators will produce NA values for the first <degree>
generations; this will be fixed in future releases.

Examples:
  --estimator allele-freq
  --estimator heritability
  --estimator 'pheno-cor(genetic)'
  --estimator 'cousin-cov(total,2)'

Quoting matters here: unescaped, the shell will try to interpret the parentheses
itself rather than passing them through to knoop, so wrap any functional-syntax
argument in quotes.
  )";

  docs[Option::VirtualValue] = R"(
--value <val(s)> supplies data inline on the command line. What is expected
depends on which flag opened the current context:

  --locus-freq / --locus-rec / --locus-mut   a single value (broadcast to every
                                              locus) or one value per locus
  --loci                                     whitespace- or comma-separated causal
                                              locus indices
  --effects                                  a single value (broadcast to every
                                              causal locus) or one value per locus
  --pheno-gen-cor / --pheno-env-cor          a symmetric P x P matrix, entered as
                                              rows separated by ';' or a newline,
                                              values within a row separated by a
                                              comma or space; a single row
                                              collapses to a vector

Example: --locus-freq --value 0.25                       (constant allele freqs)
         --pheno-gen-cor --value "1,0.5;0.5,1"            (an inline 2x2 matrix)
  )";

  docs[Option::VirtualFile] = R"(
--file <path> loads data from a file. The format is plain text: values separated
by a comma or space, rows separated by a semicolon or newline; a file with a
single row is read as a vector rather than a 1xN matrix. Tabs are not a recognised
separator.

Which flag opens the context determines what the loaded data is used for, exactly
as with --value: locus allele frequencies / recombination / mutation probabilities,
causal locus indices, effect sizes, or a phenotype gen-cor / env-cor / mate-cor matrix.
  )";

  docs[Option::VirtualDistribution] = R"(
--dist <distribution> draws the requested quantity from a probability distribution
instead of supplying it directly, e.g. --dist 'normal(0,1)'. Note that the argument
must be quoted. Supported distributions and their parameters:

  uniform(lo, hi)
  beta(a, b)
  exponential(lambda)
  gamma(shape, scale)
  normal(location, scale)
  laplace(location, scale)
  students_t(df)
  )";

  docs[Option::VirtualNumLoci] = R"(
Sets a count that depends on where it appears. At global / genome scope it sets
the number of loci simulated across the whole genome. Inside a --phenotype block,
it instead sets that phenotype's number of causal loci (defaulting to genome-loci /
<n_phenotypes> if never given).
  )";

  docs[Option::VirtualSingularValues] = R"(
Builds a correlation or cross-correlation matrix from a prescribed set of
singular values. This is done via drawing two random orthonormal matrices and
computing U * diag(<sv(s)...>) * V^T. In the event that a symmetric matrix is
required, only a single matrix is drawn, i.e., the matrix is generated as
U * diag(<sv(s)...>) * U^T.

Keep in mind that this mode of generation does not enforce any particular kind
of definiteness of the generated matrix.
  )";

  std::string subtitle = "Help for option " + option_flag;
  std::cout << std::endl << subtitle << std::endl;
  std::cout << std::string(subtitle.length(), '-');
  std::cout << docs[option] << std::endl;
}

std::vector<option> getOptions() {
  // NOLINTBEGIN(modernize-use-designated-initializers)
  return {
      // other options
      {"help", no_argument, nullptr, 'h'},
      {"version", no_argument, nullptr, 'v'},

      // global options
      {"n-individuals",
       required_argument,
       nullptr,
       Option::GlobalNumIndividuals},
      {"n-generations",
       required_argument,
       nullptr,
       Option::GlobalNumGenerations},
      {"n-replicates", required_argument, nullptr, Option::GlobalNumReplicates},
      {"n-workers", required_argument, nullptr, Option::GlobalNumWorkers},
      {"n-threads-per-worker",
       required_argument,
       nullptr,
       Option::GlobalNumThreadsPerWorker},
      {"n-threads", required_argument, nullptr, Option::GlobalNumThreads},
      {"random-seed", required_argument, nullptr, Option::GlobalRandomSeed},
      {"out-dir", required_argument, nullptr, Option::GlobalOutputDirectory},
      {"out-name", required_argument, nullptr, Option::GlobalOutputName},
      {"output-dir", required_argument, nullptr, Option::GlobalOutputDirectory},
      {"output-name", required_argument, nullptr, Option::GlobalOutputName},
      {"pedigree-max-depth",
       required_argument,
       nullptr,
       GlobalPedigreeMaxDepth},
      {"pedigree-warmup", no_argument, nullptr, Option::GlobalPedigreeWarmup},
      {"log-to-stdout", no_argument, nullptr, Option::GlobalLogNoFile},
      {"log-level", required_argument, nullptr, Option::GlobalLogLevel},
      {"save-config", no_argument, nullptr, Option::GlobalSaveConfig},
      {"load-config", required_argument, nullptr, Option::GlobalLoadConfig},
      {"share-init-state", no_argument, nullptr, Option::GlobalShareInitState},
      {"no-delete-tmp", no_argument, nullptr, Option::GlobalNoDeleteTmp},
      {"no-run", no_argument, nullptr, Option::GlobalNoRun},

      // Genome options
      {"loc-freq", no_argument, nullptr, Option::GenomeLocusInitFreqs},
      {"loc-rec", no_argument, nullptr, Option::GenomeLocusRecombinationProbs},
      {"loc-mut", no_argument, nullptr, Option::GenomeLocusMutationProbs},
      {"locus-freq", no_argument, nullptr, Option::GenomeLocusInitFreqs},
      {"locus-rec",
       no_argument,
       nullptr,
       Option::GenomeLocusRecombinationProbs},
      {"locus-mut", no_argument, nullptr, Option::GenomeLocusMutationProbs},

      // Phenotype options
      {"phenotype", required_argument, nullptr, Option::PhenotypeSpec},
      {"var-genetic", required_argument, nullptr, Option::PhenotypeVarGenetic},
      {"var-environmental",
       required_argument,
       nullptr,
       PhenotypeVarEnvironmental},
      {"var-vertical", required_argument, nullptr, PhenotypeVarVertical},
      {"effects", no_argument, nullptr, PhenotypeLocusEffects},
      {"loci", no_argument, nullptr, PhenotypeLocusIndices},
      {"pheno-gen-cor", no_argument, nullptr, PhenotypeGeneticCor},
      {"pheno-env-cor", no_argument, nullptr, PhenotypeEnvironmentalCor},
      {"pheno-genetic-cor", no_argument, nullptr, PhenotypeGeneticCor},
      {"pheno-environmental-cor",
       no_argument,
       nullptr,
       PhenotypeEnvironmentalCor},

      // Mating options
      {"mating", required_argument, nullptr, Option::MatingSpec},
      {"mate-cor", no_argument, nullptr, Option::MatingCor},
      {"tol-inf", required_argument, nullptr, Option::MatingErrorTolerance},
      {"max-itr", required_argument, nullptr, Option::MatingMaxIterations},
      {"temp-init",
       required_argument,
       nullptr,
       Option::MatingAnnealingTempInit},
      {"temp-decay",
       required_argument,
       nullptr,
       Option::MatingAnnealingTempDecay},
      {"exact", no_argument, nullptr, Option::MatingExact},

      // Population estimator options
      {"estimator",
       required_argument,
       nullptr,
       Option::PopulationEstimatorSpec},

      // Sample options
      {"sample", required_argument, nullptr, Option::SampleSpec},
      {"proband", required_argument, nullptr, Option::SampleProbandType},
      {"n-probands", required_argument, nullptr, Option::SampleNumProbands},
      {"weight", required_argument, nullptr, Option::SampleWeightFunction},
      {"on", required_argument, nullptr, Option::SampleWeightOnPhenotypes},
      {"of", required_argument, nullptr, Option::SampleWeightOfMembers},
      {"agg", required_argument, nullptr, Option::SampleWeightAggregation},
      {"include", required_argument, nullptr, Option::SampleIncludeMembers},
      {"estimators", required_argument, nullptr, Option::SampleEstimators},

      // Sample estimator options
      {"sample-estimator",
       required_argument,
       nullptr,
       Option::SampleEstimatorSpec},
      {"type", required_argument, nullptr, Option::SampleEstimatorType},
      {"exec", required_argument, nullptr, Option::SampleEstimatorExec},
      {"n-rows", required_argument, nullptr, Option::SampleEstimatorNumRows},
      {"n-cols", required_argument, nullptr, Option::SampleEstimatorNumCols},
      {"row-names",
       required_argument,
       nullptr,
       Option::SampleEstimatorRowNames},
      {"col-names",
       required_argument,
       nullptr,
       Option::SampleEstimatorColNames},

      // Virtual (context-sensitive) options
      {"n-loci", required_argument, nullptr, Option::VirtualNumLoci},
      {"file", required_argument, nullptr, Option::VirtualFile},
      {"dist", required_argument, nullptr, Option::VirtualDistribution},
      {"value", required_argument, nullptr, Option::VirtualValue},
      {"singular-values",
       required_argument,
       nullptr,
       Option::VirtualSingularValues},
      {nullptr, 0, nullptr, 0}};
  // NOLINTEND(modernize-use-designated-initializers)
}

int main(int argc, char* argv[]) {
  // get options
  std::vector<option> opts = getOptions();

  // context helps us resolve what options are legal
  Context context = Context::Global;

  // auxiliary boolean variables to manage CLI control flow
  bool run = true;
  bool save_config = false;

  // for options processing
  int opt;

  try {
    // invocation without arguments — display help and exit
    if (argc == 1) {
      displayHeader();
      displayHelpShort();
      exit(EXIT_SUCCESS);
    }

    // check for the --help flag; if specified, either display the full message
    // if no other arguments, or show option documentation for every other
    // flag named on the command line
    for (int i = 1; i < argc; ++i) {
      if (std::string_view(argv[i]) == "--help" ||
          std::string_view(argv[i]) == "-h") {
        if (argc == 2) {
          displayHeader();
          displayHelp();
          exit(EXIT_SUCCESS);
        }

        displayHeader();
        for (int j = 1; j < argc; ++j) {
          if (j == i) continue;

          std::string opt_flag = argv[j];
          auto pos = opt_flag.find_first_not_of('-');
          std::string opt_name = opt_flag.substr(pos);

          Option opt;
          bool found = false;
          for (const auto& option : opts) {
            if (option.name == nullptr) break;
            if (opt_name == "help" || opt_name == "h") {
              opt = Option::GlobalHelp;
              found = true;
              break;
            }
            if (opt_name == "version" || opt_name == "v") {
              opt = Option::GlobalVersion;
              found = true;
              break;
            }
            if (option.name == opt_name) {
              opt = static_cast<Option>(option.val);
              found = true;
              break;
            }
          }

          if (!found)
            throw std::runtime_error(
                std::format("Unrecognised option '{}'", opt_flag));

          displayHelpOption(opt, opt_flag);
        }
        exit(EXIT_SUCCESS);
      }
    }

    // set up a configuration object; read from file if specified by user
    knoop::SimulationSpec spec;

    for (int i = 1; i < argc; ++i) {
      if (std::string_view(argv[i]) == "--load-config") {
        if ((i + 1) == argc) break;
        std::filesystem::path config_path = std::string_view(argv[i + 1]);
        knoop::ConfigReader config(config_path);
        spec = config.result();
      }
    }

    // NOLINTBEGIN(bugprone-switch-missing-default-case)
    while ((opt = getopt_long(argc, argv, "hv", opts.data(), nullptr)) != -1) {
      // Default options
      switch (opt) {
        case 'v':
          displayVersion();
          exit(EXIT_SUCCESS);
        case 'h':
          // no need to do anything here since the help flag is resolved
          continue;
        case '?':
          throw std::runtime_error(
              std::format("Unrecognised option '{}'", argv[optind - 1]));
      }

      // Global options
      switch (opt) {
        case Option::GlobalNumIndividuals:
          context = Context::Global;
          spec.n_individuals =
              knoop::parse<std::size_t>(optarg, "--n-individuals");
          continue;
        case Option::GlobalNumGenerations:
          context = Context::Global;
          spec.n_generations =
              knoop::parse<std::size_t>(optarg, "--n-generations");
          continue;
        case Option::GlobalNumWorkers:
          context = Context::Global;
          spec.n_workers = knoop::parse<std::size_t>(optarg, "--n-workers");
          continue;
        case Option::GlobalNumThreadsPerWorker:
          context = Context::Global;
          spec.n_threads_per_worker =
              knoop::parse<std::size_t>(optarg, "n--threads-per-worker");
          continue;
        case Option::GlobalNumThreads:
          context = Context::Global;
          spec.n_threads = knoop::parse<std::size_t>(optarg, "--n-threads");
          continue;
        case Option::GlobalNumReplicates:
          context = Context::Global;
          spec.n_replicates =
              knoop::parse<std::size_t>(optarg, "--n-replicates");
          continue;
        case Option::GlobalRandomSeed:
          context = Context::Global;
          spec.random_seed =
              knoop::parse<std::uint64_t>(optarg, "--random-seed");
          continue;
        case Option::GlobalOutputDirectory:
          context = Context::Global;
          spec.output_dir = optarg;
          continue;
        case Option::GlobalOutputName:
          context = Context::Global;
          spec.output_name = optarg;
          continue;
        case Option::GlobalPedigreeMaxDepth:
          context = Context::Global;
          spec.pedigree_max_depth =
              knoop::parse<std::size_t>(optarg, "--pedigree-max-depth");
          continue;
        case Option::GlobalPedigreeWarmup:
          context = Context::Global;
          spec.pedigree_warmup = true;
          continue;
        case Option::GlobalLogLevel:
          context = Context::Global;
          spec.log_level = knoop::logLevelFromString(optarg);
          continue;
        case Option::GlobalLogNoFile:
          context = Context::Global;
          spec.log_to_file = false;
          continue;
        case Option::GlobalSaveConfig:
          context = Context::Global;
          save_config = true;
          continue;
        case Option::GlobalLoadConfig:
          // load config is loaded in before any processing begins, so this
          // so this is left empty
          continue;
        case Option::GlobalShareInitState:
          context = Context::Global;
          spec.share_init_state = true;
          continue;
        case Option::GlobalNoDeleteTmp:
          context = Context::Global;
          spec.delete_tmp = false;
          continue;
        case Option::GlobalNoRun:
          context = Context::Global;
          run = false;
          continue;
      }

      // Genome configuration
      switch (opt) {
        case Option::GenomeLocusInitFreqs:
          context = Context::GenomeInitAlleleFreqs;
          continue;
        case Option::GenomeLocusRecombinationProbs:
          context = Context::GenomeRecombinationProbs;
          continue;
        case Option::GenomeLocusMutationProbs:
          context = Context::GenomeMutationProbs;
          continue;
      }

      // Phenome configuration
      switch (opt) {
        case Option::PhenotypeSpec: {
          context = Context::Phenotype;
          knoop::Phenotype pheno_new{.name = optarg};
          spec.phenotypes.push_back(pheno_new);
          continue;
        }
        case Option::PhenotypeVarGenetic:
          checkContext(context, Context::Phenotype, "--var-genetic");
          spec.phenotypes.back().var_genetic =
              knoop::parse<double>(optarg, "--var-genetic");
          continue;
        case Option::PhenotypeVarEnvironmental:
          checkContext(context, Context::Phenotype, "--var-environmental");
          spec.phenotypes.back().var_environmental =
              knoop::parse<double>(optarg, "--var-environmental");
          continue;
        case Option::PhenotypeVarVertical:
          checkContext(context, Context::Phenotype, "--var-vertical");
          spec.phenotypes.back().var_vertical =
              knoop::parse<double>(optarg, "--var-vertical");
          continue;
        case Option::PhenotypeGeneticCor:
          context = Context::PhenotypeGeneticCorrelation;
          continue;
        case Option::PhenotypeEnvironmentalCor:
          context = Context::PhenotypeEnvironmentalCorrelation;
          continue;
        case Option::PhenotypeLocusEffects:
          checkContext(context, Context::Phenotype, "--effects");
          context = Context::PhenotypeEffectSizes;
          continue;
        case Option::PhenotypeLocusIndices:
          checkContext(context, Context::Phenotype, "--loci");
          context = Context::PhenotypeCausalLoci;
          continue;
      }

      // Mating configuration
      switch (opt) {
        case Option::MatingSpec:
          if (std::string_view(optarg) == "random") {
            spec.mating.type = "random";
          } else if (std::string_view(optarg) == "assortative") {
            context = Context::Mating;
            spec.mating.type = "assortative";
          } else
            throw std::invalid_argument(
                "Unsupported mating type " + std::string(optarg));
          continue;
        case Option::MatingCor:
          checkContext(context, Context::Mating, "--mate-cor");
          context = Context::MatingCorrelation;
          continue;
        case Option::MatingErrorTolerance:
          checkContext(context, Context::Mating, "--tol-inf");
          spec.mating.tolerance = knoop::parse<double>(optarg, "--tol-inf");
          continue;
        case Option::MatingMaxIterations:
          checkContext(context, Context::Mating, "--max-itr");
          spec.mating.max_iterations =
              knoop::parse<std::size_t>(optarg, "--max-itr");
          continue;
        case Option::MatingAnnealingTempInit:
          checkContext(context, Context::Mating, "--temp-init");
          spec.mating.initial_temperature =
              knoop::parse<double>(optarg, "--temp-init");
          continue;
        case Option::MatingAnnealingTempDecay:
          checkContext(context, Context::Mating, "--temp-decay");
          spec.mating.temperature_decay =
              knoop::parse<double>(optarg, "--temp-decay");
          continue;
        case Option::MatingExact:
          checkContext(context, Context::Mating, "--exact");
          spec.mating.exact = true;
          continue;
      }

      // Estimators and sampling
      switch (opt) {
        case Option::PopulationEstimatorSpec: {
          context = Context::PopulationEstimator;

          auto [name, params] = knoop::parseFunction(optarg);
          auto estimator = knoop::buildPopulationEstimator(name, params);
          estimator.name = optarg;
          spec.estimators.push_back(estimator);

          continue;
        }
        case Option::SampleEstimatorSpec: {
          context = Context::SampleEstimator;
          knoop::SampleEstimatorSpec sample_est_desc{.name = optarg};
          spec.sample_estimator_spec.push_back(sample_est_desc);
          continue;
        }
        case Option::SampleEstimatorType: {
          checkContext(context, Context::SampleEstimator, "--type");
          auto [name, params] = knoop::parseFunction(optarg);
          spec.sample_estimator_spec.back().type = name;
          if (!params.empty())
            spec.sample_estimator_spec.back().params = params;
          continue;
        }
        case Option::SampleEstimatorExec: {
          checkContext(context, Context::SampleEstimator, "--exec");

          // WRDE_NOCMD flag disallows command substitution, important for e.g.
          // R users which may want to use $ or `` in their exec string
          wordexp_t exp;
          int rc = wordexp(optarg, &exp, WRDE_NOCMD);
          if (rc != 0)
            throw std::runtime_error(
                std::format(
                    "Failed to parse --exec argument '{}' (code {})",
                    optarg,
                    rc));

          std::vector<std::string> tokens(
              exp.we_wordv, exp.we_wordv + exp.we_wordc);

          wordfree(&exp);

          if (tokens.empty())
            throw std::invalid_argument("--exec requires a command");

          spec.sample_estimator_spec.back().exec = tokens.front();
          spec.sample_estimator_spec.back().exec_args =
              std::vector<std::string>(tokens.begin() + 1, tokens.end());
          continue;
        }
        case Option::SampleEstimatorNumRows:
          checkContext(context, Context::SampleEstimator, "--n-rows");
          spec.sample_estimator_spec.back().n_rows =
              knoop::parse<std::size_t>(optarg, "--n-rows");
          continue;
        case Option::SampleEstimatorNumCols:
          checkContext(context, Context::SampleEstimator, "--n-cols");
          spec.sample_estimator_spec.back().n_cols =
              knoop::parse<std::size_t>(optarg, "--n-cols");
          continue;
        case Option::SampleEstimatorRowNames:
          checkContext(context, Context::SampleEstimator, "--row-names");
          spec.sample_estimator_spec.back().row_names =
              knoop::utils::splitString(optarg);
          continue;
        case Option::SampleEstimatorColNames:
          checkContext(context, Context::SampleEstimator, "--col-names");
          spec.sample_estimator_spec.back().col_names =
              knoop::utils::splitString(optarg);
          continue;
        case Option::SampleSpec: {
          context = Context::Sample;
          knoop::SampleSpec sample_desc;
          sample_desc.name = optarg;
          spec.sample_spec.push_back(sample_desc);
          continue;
        }
        case Option::SampleProbandType:
          checkContext(context, Context::Sample, "--proband");
          spec.sample_spec.back().proband_type = optarg;
          continue;
        case Option::SampleNumProbands:
          checkContext(context, Context::Sample, "--n-probands");
          spec.sample_spec.back().n_probands =
              knoop::parse<std::size_t>(optarg, "--n-probands");
          continue;
        case Option::SampleWeightOnPhenotypes: {
          checkContext(context, Context::Sample, "--on");
          spec.sample_spec.back().on = optarg;
          continue;
        }
        case Option::SampleWeightOfMembers:
          checkContext(context, Context::Sample, "--of");
          spec.sample_spec.back().of = optarg;
          continue;
        case Option::SampleWeightAggregation:
          checkContext(context, Context::Sample, "--agg");
          spec.sample_spec.back().agg = optarg;
          continue;
        case Option::SampleWeightFunction:
          checkContext(context, Context::Sample, "--weight");
          spec.sample_spec.back().weight_function = optarg;
          continue;
        case Option::SampleIncludeMembers:
          checkContext(context, Context::Sample, "--include");
          spec.sample_spec.back().include = optarg;
          continue;
        case Option::SampleEstimators:
          checkContext(context, Context::Sample, "--estimators");
          spec.sample_spec.back().estimators =
              knoop::utils::splitString(optarg);
          continue;
      }

      // Context-sensitive (virtual) operation flags
      Context domain = contextDomain(context);
      switch (opt) {
        case Option::VirtualNumLoci: {
          auto n_loc = knoop::parse<std::size_t>(optarg, "--n-loci");
          if (domain == Context::Phenotype)
            spec.phenotypes.back().n_causal_loci = n_loc;
          else
            spec.genome.n_loci = n_loc;
          continue;
        }
        case Option::VirtualValue: {
          if (context == PhenotypeCausalLoci) {
            auto value =
                knoop::parse<std::vector<std::size_t>>(optarg, "--value");
            spec.phenotypes.back().causal_loci = value;
            continue;
          }

          auto matrix = knoop::parse<Eigen::MatrixXd>(optarg, "--value");
          if (context == Context::GenomeInitAlleleFreqs) {
            if (matrix.size() != 1)
              throw std::invalid_argument(
                  "--locus-freq takes a single constant with --value");
            spec.genome.locus_freq = matrix(0);
          }
          if (context == Context::GenomeRecombinationProbs) {
            if (matrix.size() != 1)
              throw std::invalid_argument(
                  "--locus-rec takes a single constant with --value");
            spec.genome.locus_rec = matrix(0);
          }
          if (context == Context::GenomeMutationProbs) {
            if (matrix.size() != 1)
              throw std::invalid_argument(
                  "--locus-mut takes a single constant with --value");
            spec.genome.locus_mut = matrix(0);
          }
          if (context == Context::PhenotypeEffectSizes) {
            if (matrix.size() != 1)
              throw std::invalid_argument(
                  "--effects takes a single constant with --value");
            spec.phenotypes.back().effects = matrix(0);
          }
          if (context == Context::PhenotypeGeneticCorrelation)
            spec.genetic_component_cor = matrix;
          if (context == Context::PhenotypeEnvironmentalCorrelation)
            spec.environmental_component_cor = matrix;
          if (context == Context::MatingCorrelation)
            spec.mating.mate_cor = matrix;
          continue;
        }
        case Option::VirtualFile: {
          if (context == PhenotypeCausalLoci) {
            auto file = knoop::File<std::vector<std::size_t>>{.path = optarg};
            spec.phenotypes.back().causal_loci = file;
            continue;
          }
          auto file = knoop::File<Eigen::MatrixXd>{.path = optarg};
          if (context == Context::GenomeInitAlleleFreqs)
            spec.genome.locus_freq = file;
          if (context == Context::GenomeRecombinationProbs)
            spec.genome.locus_rec = file;
          if (context == Context::GenomeMutationProbs)
            spec.genome.locus_mut = file;
          if (context == Context::PhenotypeEffectSizes)
            spec.phenotypes.back().effects = file;
          if (context == Context::PhenotypeGeneticCorrelation)
            spec.genetic_component_cor = file;
          if (context == Context::PhenotypeEnvironmentalCorrelation)
            spec.environmental_component_cor = file;
          if (context == Context::MatingCorrelation)
            spec.mating.mate_cor = file;
          continue;
        }
        case Option::VirtualSingularValues: {
          auto values =
              knoop::parse<std::vector<double>>(optarg, "--singular-values");
          if (context == Context::PhenotypeGeneticCorrelation) {
            spec.genetic_component_cor =
                knoop::utils::matrixFromSingularValues(values, true);
          }
          if (context == Context::PhenotypeEnvironmentalCorrelation) {
            spec.environmental_component_cor =
                knoop::utils::matrixFromSingularValues(values, true);
          }
          if (context == Context::MatingCorrelation) {
            spec.mating.mate_cor =
                knoop::utils::matrixFromSingularValues(values);
          }
          continue;
        }
        case Option::VirtualDistribution: {
          auto dist = knoop::parse<knoop::Distribution>(optarg, "--dist");

          if (context == Context::GenomeInitAlleleFreqs)
            spec.genome.locus_freq = dist;
          if (context == Context::GenomeRecombinationProbs)
            spec.genome.locus_rec = dist;
          if (context == Context::GenomeMutationProbs)
            spec.genome.locus_mut = dist;
          if (context == Context::PhenotypeEffectSizes)
            spec.phenotypes.back().effects = dist;
          continue;
        }
      }
    }
    // NOLINTEND(bugprone-switch-missing-default-case)

    // resolve the seed here so the value lands in the spec, and therefore in
    // a saved config, rather than being invented downstream
    spec.random_seed = knoop::random::Xoshiro256pp::setSeed(spec.random_seed);

    if (save_config) knoop::ConfigWriter config(spec);
    if (run) knoop::runSimulation(spec);
    exit(EXIT_SUCCESS);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}
