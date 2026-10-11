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
#include <knoop/core/state.h>
#include <omp.h>

namespace knoop {

// Class to initialise founder population genotypes from unlinked loci
class GenerateHaplotypes {
 public:
  explicit GenerateHaplotypes(const Params& params)
      : random_seed_(params.global.random_seed),
        loc_maf_(params.geno.locus_freq) {};

  void operator()(State& state);

 private:
  const std::uint64_t random_seed_;
  const Eigen::VectorXd& loc_maf_;
};

inline void GenerateHaplotypes::operator()(State& state) {
  GenotypeBuffer& buf = state.geno();

  if (buf.view() != BufferLayout::LocusMajor)
    throw std::runtime_error("generate haplotypes in loc-major view");

  HaplotypeBuffer& h0 = buf.paternal();
  HaplotypeBuffer& h1 = buf.maternal();
  std::size_t n_ind = h0.individuals();
  std::size_t n_loc = buf.loci();
  std::size_t n_words = buf.words();

#pragma omp parallel
  {
    boost::random::xoshiro256pp& engine = random::Xoshiro256pp::getInstance();
    random::Xoshiro256pp::setSeed(engine, random_seed_);

#pragma omp for schedule(static)
    for (std::size_t loc = 0; loc < n_loc; ++loc) {
      // set probability of bernoulli generator
      const std::uint16_t thresh_loc = random::probThreshold(loc_maf_(loc));

      // row pointers for easy access
      std::uint64_t* word0 = h0.rowptr(loc);
      std::uint64_t* word1 = h1.rowptr(loc);

      for (std::size_t word = 0; word < n_words; ++word) {
        word0[word] = random::bernoulliWord(engine, thresh_loc);
        word1[word] = random::bernoulliWord(engine, thresh_loc);
      }

      /**
       * if the population size is not a multiple of 64, the last padding bits
       * are filled and may cause deviatons for small configurations, so we clip
       * them out via a zero mask
       */
      if (n_ind % 64) {
        std::uint64_t mask = (1ULL << (n_ind % 64)) - 1ULL;
        word0[n_words - 1] &= mask;
        word1[n_words - 1] &= mask;
      }
    }
  }
}

}  // namespace knoop
