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

#include <knoop/core/params.h>
#include <knoop/core/random.h>
#include <knoop/core/state.h>
#include <omp.h>

#include <cstdint>
#include <vector>

namespace knoop {

namespace details {

inline std::uint64_t sumMod2(std::uint64_t par) {
  par ^= par << 1;
  par ^= par << 2;
  par ^= par << 4;
  par ^= par << 8;
  par ^= par << 16;
  par ^= par << 32;
  return par;
}

}  // namespace details

class UpdateGenome {
 public:
  explicit UpdateGenome(const Params& params)
      : n_loc_(params.geno.n_loc),
        n_ind_(params.global.n_ind),
        n_sex_(n_ind_ / 2),
        thr_rec_(((n_loc_ + 63) / 64) * 64),
        random_seed_(
            params.global.post_init_seed.value_or(params.global.random_seed)),
        max_mut_(params.geno.locus_mut.maxCoeff()),
        mut_gap_(
            (params.geno.locus_mut.maxCoeff() > 0)
                ? std::optional<random::Geometric>{random::Geometric(max_mut_)}
                : std::nullopt) {
    random::probsToThreshold(params.geno.locus_rec, thr_rec_.data());
    generateMutationAcceptMap(params.geno.locus_mut);
  };

  void operator()(State& state);

 private:
  constexpr static std::size_t kNumPerWord = 64;

  const std::size_t n_loc_;
  const std::size_t n_ind_;
  const std::size_t n_sex_;
  std::vector<std::uint16_t> thr_rec_;

  const std::uint64_t random_seed_;

  const double max_mut_;
  std::optional<random::Geometric> mut_gap_;
  std::vector<double> mut_accept_;

  void generateMutationAcceptMap(const Eigen::VectorXd& loc_mut);

  std::uint64_t recombine(
      boost::random::xoshiro256pp& engine,
      std::uint64_t paternal,
      std::uint64_t maternal,
      bool& par0,
      std::size_t word,
      std::size_t valid = 64);

  void mutate(
      boost::random::xoshiro256pp& engine,
      HaplotypeBuffer& hap,
      std::size_t ind);

  void generateOffspringHaplotypes(
      boost::random::xoshiro256pp& engine,
      State& state,
      const Individual& father,
      const Individual& mother);

  void updateGenome(State& state);
};

inline void UpdateGenome::generateMutationAcceptMap(
    const Eigen::VectorXd& loc_mut) {
  if (loc_mut.isConstant(max_mut_, 0)) return;

  mut_accept_.resize(n_loc_);
  for (std::size_t loc = 0; loc < n_loc_; ++loc)
    mut_accept_[loc] = loc_mut[loc] / max_mut_;
}

inline std::uint64_t UpdateGenome::recombine(
    boost::random::xoshiro256pp& engine,
    std::uint64_t paternal,
    std::uint64_t maternal,
    bool& par0,
    std::size_t word,
    std::size_t valid) {
  std::uint64_t recomb_mask = details::sumMod2(
      random::bernoulliWord(
          engine, thr_rec_.data() + (word * kNumPerWord), valid));
  if (par0) recomb_mask = ~recomb_mask;
  par0 = static_cast<bool>((recomb_mask >> (valid - 1)) & 1ULL);
  return ((recomb_mask & paternal) | (~recomb_mask & maternal));
}

inline void UpdateGenome::mutate(
    boost::random::xoshiro256pp& engine,
    HaplotypeBuffer& hap,
    std::size_t ind) {
  if (!mut_gap_.has_value()) return;

  for (std::size_t loc = mut_gap_->sample(engine); loc < n_loc_;
       loc += 1 + mut_gap_->sample(engine)) {
    double rand = random::Uniform::sample(engine);
    if (mut_accept_.empty() || rand < mut_accept_[loc])
      hap(ind, loc / kNumPerWord) ^= 1ULL << (loc % kNumPerWord);
  }
}

inline void UpdateGenome::generateOffspringHaplotypes(
    boost::random::xoshiro256pp& engine,
    State& state,
    const Individual& father,
    const Individual& mother) {
  const std::size_t n_words = state.geno().words();

  /**
   * each parent contributes a haplotype whose starting parity (i.e., the
   * grandparental homologue the HMM traversal starts from) is generated
   * uniformly at random
   */
  auto
      [parity_father_son,
       parity_mother_son,
       parity_father_daughter,
       parity_mother_daughter] = random::coinflips<4>(engine);

  // TODO: replace these buffer references with Individual accessors
  HaplotypeBuffer& h0 = state.geno().paternal();
  HaplotypeBuffer& h1 = state.geno().maternal();

  // offspring overwrite the grandparents' data
  HaplotypeBuffer& h0_off = state.geno(Generation::Parents).paternal();
  HaplotypeBuffer& h1_off = state.geno(Generation::Parents).maternal();

  std::size_t valid = kNumPerWord;
  for (std::size_t word = 0; word < n_words; ++word) {
    if (word == n_words - 1)
      valid = (n_loc_ % kNumPerWord == 0) ? kNumPerWord : n_loc_ % kNumPerWord;

    // son's haplotypes – son assumes the index of his father
    h0_off(father.index, word) = recombine(
        engine,
        h0(father.index, word),
        h1(father.index, word),
        parity_father_son,
        word,
        valid);

    h1_off(father.index, word) = recombine(
        engine,
        h0(mother.index, word),
        h1(mother.index, word),
        parity_mother_son,
        word,
        valid);

    // daughter's haplotypes – daughter assumes the index of her mother
    h0_off(mother.index, word) = recombine(
        engine,
        h0(father.index, word),
        h1(father.index, word),
        parity_father_daughter,
        word,
        valid);

    h1_off(mother.index, word) = recombine(
        engine,
        h0(mother.index, word),
        h1(mother.index, word),
        parity_mother_daughter,
        word,
        valid);
  }

  // mutate son
  mutate(engine, h0_off, father.index);
  mutate(engine, h1_off, father.index);

  // mutate daughter
  mutate(engine, h0_off, mother.index);
  mutate(engine, h1_off, mother.index);
}

inline void UpdateGenome::operator()(State& state) {
  if (state.geno().view() != BufferLayout::IndividualMajor)
    throw std::runtime_error("update genome requires ind-major view");

  // parallelise via pragma
  auto spouse_pairs = state.pedigree.getSpousePairs(0);

#pragma omp parallel
  {
    boost::random::xoshiro256pp& engine = random::Xoshiro256pp::getInstance();
    random::Xoshiro256pp::setSeed(engine, random_seed_, 1 + state.gen);

#pragma omp for schedule(static)
    for (std::size_t pair = 0; pair < n_sex_; ++pair) {
      auto [father, mother] = spouse_pairs[pair];
      generateOffspringHaplotypes(engine, state, father, mother);
    }
  }  // #pragma omp parallel
}

}  // namespace knoop
