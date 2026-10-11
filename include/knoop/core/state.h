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

#include <knoop/core/generation.h>
#include <knoop/core/params.h>
#include <knoop/data/genotype.h>
#include <knoop/data/matching.h>
#include <knoop/data/pedigree.h>
#include <knoop/data/phenome.h>

namespace knoop {

// State holds the current state of our simulation
struct State {
  const std::size_t n_sex;
  std::size_t gen = 0;
  std::size_t rep = 0;
  std::size_t parity = 0;

  // store current and parent generation
  std::array<GenotypeBuffer, 2> genos;
  std::array<PhenotypeBuffer, 2> phenos;
  std::array<Matching, 2> matchings;
  std::array<Matching, 2> inv_matchings;

  // for storing family data
  Pedigree pedigree;

  std::size_t getParity(Generation generation) const {
    return (generation == Generation::Current) ? parity : 1 - parity;
  }

  GenotypeBuffer& geno(Generation generation = Generation::Current) {
    return genos[getParity(generation)];
  }

  PhenotypeBuffer& pheno(Generation generation = Generation::Current) {
    return phenos[getParity(generation)];
  }

  Matching& matching(Generation generation = Generation::Current) {
    return matchings[getParity(generation)];
  }

  Matching& invMatching(Generation generation = Generation::Current) {
    return inv_matchings[getParity(generation)];
  }

  const GenotypeBuffer& geno(
      Generation generation = Generation::Current) const {
    return genos[getParity(generation)];
  }

  const PhenotypeBuffer& pheno(
      Generation generation = Generation::Current) const {
    return phenos[getParity(generation)];
  }

  const Matching& matching(Generation generation = Generation::Current) const {
    return matchings[getParity(generation)];
  }

  const Matching& invMatching(
      Generation generation = Generation::Current) const {
    return inv_matchings[getParity(generation)];
  }

  void transpose() {
    this->geno(Generation::Current).transpose();
    this->geno(Generation::Parents).transpose();
  }

  void advance() {
    gen += 1;
    parity = gen % 2;
  }

  void updatePedigree() { pedigree.push(matching(), invMatching()); }
};

inline State buildState(const Params& params) {
  return State{
      .n_sex = params.global.n_ind / 2,
      .rep = params.global.rep_id,
      .genos = {GenotypeBuffer(params), GenotypeBuffer(params)},
      .phenos = {PhenotypeBuffer(params), PhenotypeBuffer(params)},
      .matchings =
          {std::vector<std::size_t>(params.global.n_ind / 2),
           std::vector<std::size_t>(params.global.n_ind / 2)},
      .inv_matchings =
          {std::vector<std::size_t>(params.global.n_ind / 2),
           std::vector<std::size_t>(params.global.n_ind / 2)},
      .pedigree =
          Pedigree(params.global.pedigree_max_depth, params.global.n_ind)};
}

}  // namespace knoop
