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

#define SOL_ALL_SAFETIES_ON 1
#include <knoop/bindings/eigen.h>
#include <knoop/bindings/random.h>
#include <knoop/core/params.h>
#include <knoop/estimate/external.h>
#include <knoop/estimate/genome_estimators.h>
#include <knoop/estimate/greml.h>
#include <knoop/estimate/gwas.h>
#include <knoop/estimate/haseman_elston.h>
#include <knoop/estimate/mating_estimators.h>
#include <knoop/estimate/pedigree_estimators.h>
#include <knoop/estimate/phenome_estimators.h>
#include <knoop/estimate/sample.h>
#include <knoop/simulation/run.h>

#include <sol/sol.hpp>
#include <utility>

namespace knoop::lua {

template <typename T>
sol::object getOrObject(
    const sol::table& tbl, const std::string& key, const T& default_value) {
  if (tbl[key].valid()) return tbl.get<sol::object>(key);

  return sol::make_object(tbl.lua_state(), default_value);
}

inline Eigen::VectorXd solToVector(sol::object& value, std::size_t n) {
  if (value.is<double>())
    return Eigen::VectorXd::Constant(n, value.as<double>());
  if (value.is<Eigen::VectorXd>()) return value.as<Eigen::VectorXd>();
  if (value.is<sol::function>()) {
    sol::function dist = value.as<sol::function>();
    Eigen::VectorXd draw = dist(n);
    return draw;
  }
  if (value.is<std::string>()) {
    auto parsed = parseFile<Eigen::VectorXd>(value.as<std::string>());
    return parsed;
  }
  throw KnoopInvalidArgumentError(
      "Value must be either a constant, a distribution, or a file");
}

/**
 * in general, it is better that we parse files inside of lua rather than the
 * whole File class, so we should add helper functions for this
 */
inline void defineGenomeParams(sol::state& lua) {
  lua.new_usertype<GenomeParams>(
      "genome",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            std::size_t n_loci = tbl.get<std::size_t>("n_loci");
            sol::object locus_freq = getOrObject(tbl, "locus_freq", 0.5);
            sol::object locus_rec = getOrObject(tbl, "locus_rec", 0.5);
            sol::object locus_mut = getOrObject(tbl, "locus_mut", 0.0);

            return GenomeParams(
                n_loci,
                solToVector(locus_freq, n_loci),
                solToVector(locus_rec, n_loci),
                solToVector(locus_mut, n_loci));
          },
          [](sol::this_state s,
             std::size_t n_loci,
             sol::optional<sol::object> locus_freq,
             sol::optional<sol::object> locus_rec,
             sol::optional<sol::object> locus_mut) {
            sol::state_view sol(s);

            if (!locus_freq.has_value())
              locus_freq = sol::make_object(sol, 0.5);
            if (!locus_rec.has_value()) locus_rec = sol::make_object(sol, 0.5);
            if (!locus_mut.has_value()) locus_mut = sol::make_object(sol, 0.0);

            return GenomeParams(
                n_loci,
                solToVector(locus_freq.value(), n_loci),
                solToVector(locus_rec.value(), n_loci),
                solToVector(locus_mut.value(), n_loci));
          }),
      "n_loci",
      sol::readonly_property([](const GenomeParams& g) { return g.numLoci(); }),
      "locus_freq",
      sol::property(
          [](const GenomeParams& g) { return g.locusFrequencies(); },
          [](GenomeParams& g, Eigen::VectorXd v) {
            if (v.size() != g.numLoci())
              throw KnoopDimensionError(
                  "Allele frequency vector must have length {} (got {})",
                  g.numLoci(),
                  v.size());
            g.locusFrequencies() = std::move(v);
          }),
      "locus_rec",
      sol::property(
          [](const GenomeParams& g) { return g.locusRecombination(); },
          [](GenomeParams& g, Eigen::VectorXd v) {
            if (v.size() != g.numLoci())
              throw KnoopDimensionError(
                  "Recombination vector must have length {} (got {})",
                  g.numLoci(),
                  v.size());
            g.locusRecombination() = std::move(v);
          }),
      "locus_mut",
      sol::property(
          [](const GenomeParams& g) { return g.locusMutation(); },
          [](GenomeParams& g, Eigen::VectorXd v) {
            if (v.size() != g.numLoci())
              throw KnoopDimensionError(
                  "Mutation vector must have length {} (got {})",
                  g.numLoci(),
                  v.size());
            g.locusMutation() = std::move(v);
          }));
}

inline void definePhenomeTypes(sol::state& lua) {
  lua.new_enum<Component>(
      "component",
      {{"total", Component::Total},
       {"genetic", Component::Genetic},
       {"environmental", Component::Environmental},
       {"indirect", Component::Indirect}});
}

inline void definePhenotype(sol::state& lua) {
  lua.new_usertype<Phenotype>(
      "phenotype",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            auto name = tbl.get<std::string>("name");
            auto n_loci = tbl.get<std::size_t>("n_loci");
            auto effects = getOrObject(tbl, "effects", random::Normal{});
            auto freq_dependence = tbl.get_or("freq_dependence", 0.0);
            auto allele_freq = tbl.get_or("allele_freq", Eigen::VectorXd{});
            double var_genetic = tbl.get_or("var_genetic", 0.5);
            double var_environmental = tbl.get_or("var_environmental", 0.5);
            double var_indirect = tbl.get_or("var_indirect", 0.0);
            double prop_indirect_paternal =
                tbl.get_or("prop_indirect_paternal", 0.5);

            return Phenotype(
                name,
                n_loci,
                solToVector(effects, n_loci),
                freq_dependence,
                allele_freq,
                /**
                 * For now, causal loci are not directly specified.
                 */
                std::nullopt,
                var_genetic,
                var_environmental,
                var_indirect,
                prop_indirect_paternal);
          },
          [](sol::this_state s,
             std::string name,
             std::size_t n_loci,
             sol::optional<sol::object> effects,
             double freq_dependence = 0.0,
             Eigen::VectorXd allele_freq = Eigen::VectorXd{},
             double var_genetic = 0.5,
             double var_environmental = 0.5,
             double var_indirect = 0.0,
             double prop_indirect_paternal = 0.0) {
            if (!effects.has_value()) {
              sol::state_view sol(s);
              effects = sol::make_object(sol, random::Normal{});
            }

            return Phenotype(
                name,
                n_loci,
                solToVector(effects.value(), n_loci),
                freq_dependence,
                allele_freq,
                /**
                 * For now, causal loci are not directly specified.
                 */
                std::nullopt,
                var_genetic,
                var_environmental,
                var_indirect,
                prop_indirect_paternal);
          }),
      "name",
      sol::property(
          [](const Phenotype& phenotype) { return phenotype.name(); },
          [](Phenotype& phenotype, const std::string& v) {
            phenotype.name() = v;
          }),
      "n_loci",
      sol::property(
          [](const Phenotype& phenotype) { return phenotype.numCausalLoci(); },
          [](Phenotype& phenotype, std::size_t v) {
            phenotype.numCausalLoci() = v;
          }),
      "effects",
      sol::property(
          [](const Phenotype& phenotype) { return phenotype.effects(); },
          [](Phenotype& phenotype, Eigen::VectorXd& v) {
            phenotype.effects() = v;
          }),
      "var_genetic",
      sol::property(
          [](const Phenotype& phenotype) { return phenotype.varGenetic(); },
          [](Phenotype& phenotype, double v) { phenotype.varGenetic() = v; }),
      "var_environmental",
      sol::property(
          [](const Phenotype& phenotype) {
            return phenotype.varEnvironmental();
          },
          [](Phenotype& phenotype, double v) {
            phenotype.varEnvironmental() = v;
          }),
      "var_indirect",
      sol::property(
          [](const Phenotype& phenotype) { return phenotype.varIndirect(); },
          [](Phenotype& phenotype, double v) { phenotype.varIndirect() = v; }),
      "prop_indirect_paternal",
      sol::property(
          [](const Phenotype& phenotype) {
            return phenotype.propIndirectPaternal();
          },
          [](Phenotype& phenotype, double v) {
            phenotype.propIndirectPaternal() = v;
          }),
      sol::meta_function::to_string,
      [](const Phenotype& pheno) {
        std::stringstream ss;
        ss << std::format(
            "Phenotype{{name = {}, genetic = {}, environmental = {}, indirect "
            "= {}}}",
            pheno.name(),
            pheno.varGenetic(),
            pheno.varEnvironmental(),
            pheno.varIndirect());
        return ss.str();
      });
}

inline void definePhenomeParams(sol::state& lua) {
  lua.new_usertype<PhenotypeParams>(
      "phenotypes",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            auto phenotypes = tbl.get<std::vector<Phenotype>>("phenotypes");

            std::size_t n_pheno = phenotypes.size();
            Eigen::MatrixXd identity =
                Eigen::MatrixXd::Identity(n_pheno, n_pheno);

            return PhenotypeParams(
                phenotypes,
                tbl.get_or("gen_cor", identity),
                tbl.get_or("env_cor", identity));
          },
          [](sol::this_state s,
             std::vector<Phenotype> phenotypes,
             sol::optional<Eigen::MatrixXd> cor_genetic,
             sol::optional<Eigen::MatrixXd> cor_environmental) {
            sol::state_view sol(s);

            std::size_t n_pheno = phenotypes.size();
            Eigen::MatrixXd identity =
                Eigen::MatrixXd::Identity(n_pheno, n_pheno);

            return PhenotypeParams(
                phenotypes,
                cor_genetic.value_or(identity),
                cor_environmental.value_or(identity));
          }),
      sol::meta_function::index,
      sol::overload(
          [](PhenotypeParams& self, std::size_t index) {
            return self.array()[index - 1];
          },
          [](PhenotypeParams& self, const std::string& name) {
            return self.array()[self.id(name)];
          }));
}

inline void defineMatingParams(sol::state& lua) {
  lua.new_enum<MatingType>(
      "mating_type",
      {{"assortative", MatingType::Assortative},
       {"random", MatingType::Random}});

  lua.new_usertype<MatingParams>(
      "mating",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            Eigen::MatrixXd default_cor{};
            return MatingParams(
                tbl.get_or("type", MatingType::Random),
                tbl.get_or("cor", default_cor),
                tbl.get_or("tol", 0.1),
                tbl.get_or("max_itr", 0UZ),
                tbl.get_or("temp_init", 1.0),
                tbl.get_or("temp_decay", 0.99),
                tbl.get_or("exact", false),
                tbl.get_or("whiten", true));
          },
          [](sol::this_state s,
             sol::optional<MatingType> type,
             sol::optional<Eigen::MatrixXd> cor,
             sol::optional<double> tol,
             sol::optional<std::size_t> max_itr,
             sol::optional<double> temp_init,
             sol::optional<double> temp_decay,
             sol::optional<bool> exact,
             sol::optional<bool> whiten) {
            Eigen::MatrixXd default_cor{};
            return MatingParams(
                type.value_or(MatingType::Random),
                cor.value_or(default_cor),
                tol.value_or(0.1),
                max_itr.value_or(0UZ),
                temp_init.value_or(1.0),
                temp_decay.value_or(0.99),
                exact.value_or(false),
                whiten.value_or(true));
          }));
}

inline void defineSamplingTypes(sol::state& lua) {
  /**
   * Define aggregators
   */
  lua.new_enum<Aggregator>(
      "aggregator",
      {{"mean", Aggregator::Mean},
       {"max", Aggregator::Max},
       {"min", Aggregator::Min},
       {"identity", Aggregator::Identity}});

  /**
   * Define weight functions – we need to wrap them in an opaque struct since
   * otherwise sol2 runs into Eigen::MatrixXd's .begin() and .end() iterators,
   * attempts to infer whether it's a container, before spontaneously
   * combusting into a million template errors
   */
  lua.new_usertype<Weight>("Weight");

  sol::table weight = lua.create_named_table("weight");

  weight["uniform"] = Weight{.weight_fn = weightUniform()};

  weight["logistic"] = [](const sol::table& tbl) {
    return Weight{
        .weight_fn = weightLogistic(tbl.get<Eigen::VectorXd>("effects"))};
  };

  weight["case_control"] = [](const sol::table& tbl) {
    return Weight{
        .weight_fn = weightCaseControl(
            tbl.get<double>("threshold"),
            tbl.get<bool>("greater"),
            tbl.get<Eigen::VectorXd>("effects"))};
  };

  lua.new_usertype<Family>(
      "family",
      // clang-format off
      "father",           sol::var(Family::Father),
      "mother",           sol::var(Family::Mother),
      "son",              sol::var(Family::Son),
      "son_wife",         sol::var(Family::SonWife),
      "daughter",         sol::var(Family::Daughter),
      "daughter_husband", sol::var(Family::DaughterHusband),
      "all",              sol::var(Family::All),
      // clang-format on
      sol::meta_function::addition,
      [](Family a, Family b) {
        return static_cast<Family>(
            std::to_underlying(a) | std::to_underlying(b));
      },
      sol::meta_function::subtraction,
      [](Family a, Family b) {
        return static_cast<Family>(
            std::to_underlying(a) & ~std::to_underlying(b));
      },
      sol::meta_function::unary_minus,
      [](Family a) {
        return static_cast<Family>(
            std::to_underlying(Family::All) & ~std::to_underlying(a));
      });

  lua.new_usertype<Mate>(
      "mate",
      // clang-format off
      "husband",                sol::var(Mate::Husband),
      "husband_father",         sol::var(Mate::HusbandFather),
      "husband_mother",         sol::var(Mate::HusbandMother),
      "husband_sister",         sol::var(Mate::HusbandSister),
      "husband_brother_in_law", sol::var(Mate::HusbandBrotherInLaw),
      "wife",                   sol::var(Mate::Wife),
      "wife_father",            sol::var(Mate::WifeFather),
      "wife_mother",            sol::var(Mate::WifeMother),
      "wife_brother",           sol::var(Mate::WifeBrother),
      "wife_sister_in_law",     sol::var(Mate::WifeSisterInLaw),
      // clang-format on
      sol::meta_function::addition,
      [](Mate a, Mate b) {
        return static_cast<Mate>(std::to_underlying(a) | std::to_underlying(b));
      },
      sol::meta_function::subtraction,
      [](Mate a, Mate b) {
        return static_cast<Mate>(
            std::to_underlying(a) & ~std::to_underlying(b));
      },
      sol::meta_function::unary_minus,
      [](Mate a) {
        return static_cast<Mate>(
            std::to_underlying(Mate::All) & ~std::to_underlying(a));
      });

  lua.new_usertype<Self>("self", "self", sol::var(Self::Self));
}

inline void defineSamples(sol::state& lua) {
  sol::table samples = lua.create_named_table("sample");

  samples["self"] = sol::overload(
      [](const sol::table& tbl) {
        sol::optional<Weight> weight_opt =
            tbl.get<sol::optional<Weight>>("weight");
        return SampleFor<ProbandType::Self>{
            .name = tbl.get<std::string>("name"),
            .n_probands = tbl.get<std::size_t>("n_probands"),
            .phenotypes_on =
                tbl.get_or<std::optional<std::vector<std::string>>>(
                    "phenotypes_on", std::nullopt),
            .phenotypes_components =
                tbl.get_or<std::optional<std::vector<Component>>>(
                    "phenotypes_components", std::nullopt),
            .members_of = tbl.get_or("of_members", Self::Self),
            .members_include = tbl.get_or("include_members", Self::Self),
            .weight = weight_opt.value_or(Weight{.weight_fn = weightUniform()}),
            .estimators =
                tbl.get_or<std::vector<SampleEstimator>>("estimators", {})};
      },
      [](std::string name,
         std::size_t n_probands,
         std::optional<std::vector<std::string>> phenotypes_on = std::nullopt,
         std::optional<std::vector<Component>> phenotypes_components =
             std::nullopt,
         Self members_of = Self::Self,
         Self members_include = Self::Self,
         Weight weight = Weight{.weight_fn = weightUniform()},
         std::vector<SampleEstimator> estimators = {}) {
        return SampleFor<ProbandType::Self>{
            .name = std::move(name),
            .n_probands = n_probands,
            .phenotypes_on = std::move(phenotypes_on),
            .phenotypes_components = std::move(phenotypes_components),
            .members_of = members_of,
            .members_include = members_include,
            .weight = std::move(weight),
            .estimators = std::move(estimators)};
      });

  samples["family"] = sol::overload(
      [](const sol::table& tbl) {
        sol::optional<Weight> weight_opt =
            tbl.get<sol::optional<Weight>>("weight");
        return SampleFor<ProbandType::Family>{
            .name = tbl.get<std::string>("name"),
            .n_probands = tbl.get<std::size_t>("n_probands"),
            .phenotypes_on =
                tbl.get_or<std::optional<std::vector<std::string>>>(
                    "phenotypes_on", std::nullopt),
            .phenotypes_components =
                tbl.get_or<std::optional<std::vector<Component>>>(
                    "phenotypes_components", std::nullopt),
            .members_of = tbl.get_or("members_of", Family::All),
            .members_include = tbl.get_or("members_include", Family::All),
            .aggregator = tbl.get_or("aggregator", Aggregator::Mean),
            .weight = weight_opt.value_or(Weight{.weight_fn = weightUniform()}),
            .estimators =
                tbl.get_or<std::vector<SampleEstimator>>("estimators", {})};
      },
      [](std::string name,
         std::size_t n_probands,
         std::optional<std::vector<std::string>> phenotypes_on = std::nullopt,
         std::optional<std::vector<Component>> phenotypes_components =
             std::nullopt,
         Family members_of = Family::All,
         Family members_include = Family::All,
         Aggregator aggregator = Aggregator::Mean,
         Weight weight = Weight{.weight_fn = weightUniform()},
         std::vector<SampleEstimator> estimators = {}) {
        return SampleFor<ProbandType::Family>{
            .name = std::move(name),
            .n_probands = n_probands,
            .phenotypes_on = std::move(phenotypes_on),
            .phenotypes_components = std::move(phenotypes_components),
            .members_of = members_of,
            .members_include = members_include,
            .aggregator = aggregator,
            .weight = std::move(weight),
            .estimators = std::move(estimators)};
      });

  samples["mate"] = sol::overload(
      [](const sol::table& tbl) {
        sol::optional<Weight> weight_opt =
            tbl.get<sol::optional<Weight>>("weight");
        return SampleFor<ProbandType::Mate>{
            .name = tbl.get<std::string>("name"),
            .n_probands = tbl.get<std::size_t>("n_probands"),
            .phenotypes_on =
                tbl.get_or<std::optional<std::vector<std::string>>>(
                    "phenotypes_on", std::nullopt),
            .phenotypes_components =
                tbl.get_or<std::optional<std::vector<Component>>>(
                    "phenotypes_components", std::nullopt),
            .members_of = tbl.get_or("members_of", Mate::Couple),
            .members_include = tbl.get_or("members_include", Mate::All),
            .aggregator = tbl.get_or("aggregator", Aggregator::Mean),
            .weight = weight_opt.value_or(Weight{.weight_fn = weightUniform()}),
            .estimators =
                tbl.get_or<std::vector<SampleEstimator>>("estimators", {})};
      },
      [](std::string name,
         std::size_t n_probands,
         std::optional<std::vector<std::string>> phenotypes_on = std::nullopt,
         std::optional<std::vector<Component>> phenotypes_components =
             std::nullopt,
         Mate members_of = Mate::Couple,
         Mate members_include = Mate::All,
         Aggregator aggregator = Aggregator::Mean,
         Weight weight = Weight{.weight_fn = weightUniform()},
         std::vector<SampleEstimator> estimators = {}) {
        return SampleFor<ProbandType::Mate>{
            .name = std::move(name),
            .n_probands = n_probands,
            .phenotypes_on = std::move(phenotypes_on),
            .phenotypes_components = std::move(phenotypes_components),
            .members_of = members_of,
            .members_include = members_include,
            .aggregator = aggregator,
            .weight = std::move(weight),
            .estimators = std::move(estimators)};
      });
}

inline void definePopulationEstimators(sol::state& lua) {
  sol::table estimators = lua.create_named_table("estimator");

  /**
   * Zero arguments
   */
  estimators["genotype_freq"] = &populationGenotypeFreq;
  estimators["genotype_mean"] = &populationGenotypeMean;
  estimators["genotype_var"] = &populationGenotypeVar;
  estimators["genotype_cov"] = &populationGenotypeCov;
  estimators["genotype_cor"] = &populationGenotypeCov;
  estimators["allele_freq"] = &populationAlleleFreq;
  estimators["heritability"] = &populationHeritability;

  /**
   * One argument
   */
  estimators["phenotype_mean"] = sol::overload(
      [](const sol::table& tbl) {
        return populationComponentMean(
            tbl.get_or("component", Component::Total));
      },
      [](sol::optional<Component> component) {
        return populationComponentMean(component.value_or(Component::Total));
      });

  estimators["phenotype_var"] = sol::overload(
      [](const sol::table& tbl) {
        return populationComponentVar(
            tbl.get_or("component", Component::Total));
      },
      [](sol::optional<Component> component) {
        return populationComponentVar(component.value_or(Component::Total));
      });

  estimators["phenotype_cov"] = sol::overload(
      [](const sol::table& tbl) {
        return populationComponentCov(
            tbl.get_or("component", Component::Total));
      },
      [](sol::optional<Component> component) {
        return populationComponentCov(component.value_or(Component::Total));
      });

  estimators["spouse_cor"] = sol::overload(
      [](const sol::table& tbl) {
        return populationSpouseCor(tbl.get_or("component", Component::Total));
      },
      [](sol::optional<Component> component) {
        return populationSpouseCor(component.value_or(Component::Total));
      });

  estimators["sibling_cov"] = sol::overload(
      [](const sol::table& tbl) {
        return populationSiblingCov(tbl.get_or("component", Component::Total));
      },
      [](sol::optional<Component> component) {
        return populationSiblingCov(component.value_or(Component::Total));
      });

  estimators["sibling_cor"] = sol::overload(
      [](const sol::table& tbl) {
        return populationSiblingCor(tbl.get_or("component", Component::Total));
      },
      [](sol::optional<Component> component) {
        return populationSiblingCor(component.value_or(Component::Total));
      });

  /**
   * Two arguments
   */
  estimators["cousin_cov"] = sol::overload(
      [](const sol::table& tbl) {
        return populationCousinCov(
            tbl.get_or("degree", 1UZ),
            tbl.get_or("component", Component::Total));
      },
      [](sol::optional<std::size_t> degree,
         sol::optional<Component> component) {
        return populationCousinCov(
            degree.value_or(1UZ), component.value_or(Component::Total));
      });

  estimators["cousin_cor"] = sol::overload(
      [](const sol::table& tbl) {
        return populationCousinCor(
            tbl.get_or("degree", 1UZ),
            tbl.get_or("component", Component::Total));
      },
      [](sol::optional<std::size_t> degree,
         sol::optional<Component> component) {
        return populationCousinCor(
            degree.value_or(1UZ), component.value_or(Component::Total));
      });

  estimators["ancestor_cov"] = sol::overload(
      [](const sol::table& tbl) {
        return populationAncestorCov(
            tbl.get_or("degree", 1UZ),
            tbl.get_or("component", Component::Total));
      },
      [](sol::optional<std::size_t> degree,
         sol::optional<Component> component) {
        return populationAncestorCov(
            degree.value_or(1UZ), component.value_or(Component::Total));
      });

  estimators["ancestor_cor"] = sol::overload(
      [](const sol::table& tbl) {
        return populationAncestorCor(
            tbl.get_or("degree", 1UZ),
            tbl.get_or("component", Component::Total));
      },
      [](sol::optional<std::size_t> degree,
         sol::optional<Component> component) {
        return populationAncestorCor(
            degree.value_or(1UZ), component.value_or(Component::Total));
      });
}

inline void defineSampleEstimators(sol::state& lua) {
  sol::table estimators = lua.create_named_table("sample_estimator");

  estimators["mean"] = &sampleMean;
  estimators["var"] = &sampleVar;
  estimators["cov"] = &sampleCov;
  estimators["spouse_cor"] = &sampleSpouseCor;

  estimators["greml"] = sol::overload(
      [](const sol::table& tbl) {
        return sampleGREML(tbl.get_or("name", std::string("greml")));
      },
      [](sol::optional<std::string> name) {
        return sampleGREML(name.value_or("greml"));
      });

  estimators["haseman_elston"] = sol::overload(
      [](const sol::table& tbl) {
        return sampleHasemanElston(
            tbl.get_or("name", std::string("haseman-elston")));
      },
      [](sol::optional<std::string> name) {
        return sampleHasemanElston(name.value_or("haseman-elston"));
      });

  estimators["gwas"] = sol::overload(
      [](const sol::table& tbl) {
        return sampleGWAS(
            tbl.get_or("name", std::string("gwas")),
            tbl.get_or("n_pcs", 0UZ),
            tbl.get_or("pval_threshold", 5e-8));
      },
      [](sol::optional<std::string> name,
         sol::optional<std::size_t> n_pcs,
         sol::optional<double> pval_threshold) {
        return sampleGWAS(
            name.value_or("gwas"),
            n_pcs.value_or(0UZ),
            pval_threshold.value_or(5e-8));
      });

  estimators["external"] = sol::overload(
      [](const sol::table& tbl) {
        return sampleExternal(
            tbl.get<std::string>("name"),
            tbl.get<std::string>("exec"),
            tbl.get_or<std::vector<std::string>>("exec_args", {}),
            tbl.get<std::size_t>("n_rows"),
            tbl.get_or("n_cols", 1UZ),
            tbl.get<std::optional<std::vector<std::string>>>("row_names"),
            tbl.get<std::optional<std::vector<std::string>>>("col_names"));
      },
      [](std::string name,
         std::string exec,
         sol::optional<std::vector<std::string>> exec_args,
         std::size_t n_rows,
         sol::optional<std::size_t> n_cols,
         sol::optional<std::vector<std::string>> row_names,
         sol::optional<std::vector<std::string>> col_names) {
        std::optional<std::vector<std::string>> std_row_names =
            row_names.has_value()
                ? std::optional<std::vector<std::string>>(row_names.value())
                : std::nullopt;
        std::optional<std::vector<std::string>> std_col_names =
            col_names.has_value()
                ? std::optional<std::vector<std::string>>(col_names.value())
                : std::nullopt;

        return sampleExternal(
            name,
            exec,
            exec_args.value_or(std::vector<std::string>{}),
            n_rows,
            n_cols.value_or(1UZ),
            std_row_names,
            std_col_names);
      });
}

inline void defineSimulation(sol::state& lua) {
  lua.new_enum<LogLevel>(
      "log_level",
      {{"debug", LogLevel::Debug},
       {"info", LogLevel::Info},
       {"warning", LogLevel::Warning},
       {"error", LogLevel::Error},
       {"none", LogLevel::None}});

  lua.set_function("simulation", [](const sol::table& tbl) {
    MatingParams default_mate;

    Params params(
        tbl.get<std::size_t>("n_individuals"),
        tbl.get<std::size_t>("n_generations"),
        tbl.get<GenomeParams>("genome"),
        tbl.get<PhenotypeParams>("phenotypes"),
        tbl.get_or<MatingParams>("mating", default_mate),
        tbl.get_or<std::vector<PopulationEstimator>>("estimators", {}),
        tbl.get_or<std::vector<Sample>>("samples", {}),
        tbl.get_or("n_replicates", 1UZ),
        tbl.get_or("n_workers", 1UZ),
        tbl.get_or("n_threads_per_worker", 1UZ),
        tbl.get_or("share_init_state", false),
        tbl.get_or("pedigree_warmup", false),
        tbl.get_or("pedigree_max_depth", 1UZ),
        tbl.get_or("random_seed", random::Xoshiro256pp::setSeed()),
        tbl.get_or<std::string>("output_dir", "."),
        tbl["output_name"].valid()
            ? std::optional<std::string>(tbl.get<std::string>("output_name"))
            : std::nullopt,
        tbl.get_or("log_level", LogLevel::Info),
        tbl.get_or("log_to_file", true),
        tbl.get_or("delete_temporaries", true));

    knoop::runSimulation(params);
  });
}

inline sol::state getLuaState() {
  sol::state sol;

  sol.open_libraries(sol::lib::base, sol::lib::math, sol::lib::table);

  knoop::lua::defineDistributions(sol);
  knoop::lua::defineEigenBindings(sol);
  knoop::lua::defineGenomeParams(sol);
  knoop::lua::definePhenotype(sol);
  knoop::lua::definePhenomeTypes(sol);
  knoop::lua::definePhenomeParams(sol);
  knoop::lua::defineMatingParams(sol);
  knoop::lua::definePopulationEstimators(sol);
  knoop::lua::defineSamplingTypes(sol);
  knoop::lua::defineSampleEstimators(sol);
  knoop::lua::defineSamples(sol);
  knoop::lua::defineSimulation(sol);

  return sol;
}

inline void runKnoopScript(
    sol::state& sol, const std::filesystem::path& script_path) {
  auto result = sol.safe_script_file(script_path, sol::script_pass_on_error);

  if (!result.valid()) {
    sol::error err = result;
    std::string err_msg = err.what();

    if (auto pos = err_msg.find("\nstack traceback:"); pos != std::string::npos)
      err_msg.erase(pos);

    std::cerr << std::format("knoop: error: {}\n", err_msg);
  }
}

}  // namespace knoop::lua
