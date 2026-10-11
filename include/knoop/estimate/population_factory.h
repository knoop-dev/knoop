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

#include <knoop/estimate/estimator.h>
#include <knoop/estimate/genome_estimators.h>
#include <knoop/estimate/mating_estimators.h>
#include <knoop/estimate/pedigree_estimators.h>
#include <knoop/estimate/phenome_estimators.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace knoop {

using PopulationEstimatorFactory =
    std::function<PopulationEstimator(const std::vector<std::string>&)>;

inline PopulationEstimator buildPopulationEstimator(
    const std::string& name, const std::vector<std::string>& params) {
  static const std::unordered_map<std::string, PopulationEstimatorFactory>
      Registry = {
          {"genotype-freq",
           [](const std::vector<std::string>& /*s*/) {
             return populationGenotypeFreq();
           }},
          {"genotype-mean",
           [](const std::vector<std::string>& /*s*/) {
             return populationGenotypeMean();
           }},
          {"genotype-var",
           [](const std::vector<std::string>& /*s*/) {
             return populationGenotypeVar();
           }},
          {"allele-freq",
           [](const std::vector<std::string>& /*s*/) {
             return populationAlleleFreq();
           }},
          {"genotype-cov",
           [](const std::vector<std::string>& /*s*/) {
             return populationGenotypeCov();
           }},
          {"genotype-cor",
           [](const std::vector<std::string>& /*s*/) {
             return populationGenotypeCor();
           }},
          {"heritability",
           [](const std::vector<std::string>& /*s*/) {
             return populationHeritability();
           }},
          {"pheno-mean",
           [](const std::vector<std::string>& s) {
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             return populationComponentMean(component);
           }},
          {"pheno-var",
           [](const std::vector<std::string>& s) {
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             return populationComponentVar(component);
           }},
          {"pheno-cor",
           [](const std::vector<std::string>& s) {
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             return populationComponentCor(component);
           }},
          {"pheno-cov",
           [](const std::vector<std::string>& s) {
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             return populationComponentCov(component);
           }},
          {"mate-cor",
           [](const std::vector<std::string>& s) {
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             return populationMateCor(component);
           }},
          {"sibling-cov",
           [](const std::vector<std::string>& s) {
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             return populationSiblingCov(component);
           }},
          {"sibling-cor",
           [](const std::vector<std::string>& s) {
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             return populationSiblingCor(component);
           }},
          {"cousin-cov",
           [](const std::vector<std::string>& s) {
             std::size_t degree = 1;
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             if (s.size() > 1) degree = parse<std::size_t>(s[1]);

             return populationCousinCov(degree, component);
           }},
          {"cousin-cor",
           [](const std::vector<std::string>& s) {
             std::size_t degree = 1;
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             if (s.size() > 1) degree = parse<std::size_t>(s[1]);

             return populationCousinCor(degree, component);
           }},
          {"ancestor-cov",
           [](const std::vector<std::string>& s) {
             std::size_t degree = 1;
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             if (s.size() > 1) degree = parse<std::size_t>(s[1]);

             return populationAncestorCov(degree, component);
           }},
          {"ancestor-cor", [](const std::vector<std::string>& s) {
             std::size_t degree = 1;
             Component component = Component::Total;
             if (!s.empty()) component = componentFromString(s[0]);
             if (s.size() > 1) degree = parse<std::size_t>(s[1]);

             return populationAncestorCor(degree, component);
           }}};

  auto it = Registry.find(name);
  if (it == Registry.end())
    throw std::runtime_error("Could not find population estimator " + name);

  return it->second(params);
}

}  // namespace knoop
