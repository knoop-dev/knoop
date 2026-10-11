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
#include <knoop/core/utils.h>
#include <knoop/data/haplotype.h>

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace knoop {

class GenotypeBuffer {
 public:
  explicit GenotypeBuffer(const Params& params)
      : locus_mut_(std::move(params.geno.locus_mut)),
        locus_rec_(std::move(params.geno.locus_rec)),
        allele_freqs_init_(std::move(params.geno.locus_freq)),
        locus_means_(params.geno.n_loc),
        locus_vars_(params.geno.n_loc),
        allele_freqs_(params.geno.n_loc),
        genotype_freqs_(3, params.geno.n_loc),
        paternal_(params.global.n_ind, params.geno.n_loc),
        maternal_(params.global.n_ind, params.geno.n_loc) {};

  /** \brief Returns a vector of locus population means. */
  Eigen::VectorXd& locusMean() noexcept { return locus_means_; }

  /** \brief Returns a vector of locus population variances. */
  Eigen::VectorXd& locusVar() noexcept { return locus_vars_; }

  /** \brief Returns a vector of allele population frequencies. */
  Eigen::VectorXd& alleleFreq() noexcept { return allele_freqs_; }

  /** \brief Returns a vector of genotype population frequencies. */
  Eigen::MatrixXd& genotypeFreq() noexcept { return genotype_freqs_; }

  /** \brief Returns a vector of locus population means. */
  const Eigen::VectorXd& locusMean() const noexcept { return locus_means_; }

  /** \brief Returns a vector of locus population variances. */
  const Eigen::VectorXd& locusVar() const noexcept { return locus_vars_; }

  /** \brief Returns a vector of allele population frequencies. */
  const Eigen::VectorXd& alleleFreq() const noexcept { return allele_freqs_; }

  /** \brief Returns a vector of genotype population frequencies. */
  const Eigen::MatrixXd& genotypeFreq() const noexcept {
    return genotype_freqs_;
  }

  /** \brief Returns the population genotype mean of a particular locus. */
  double locusMean(std::size_t loc) const noexcept { return locus_means_(loc); }

  /** \brief Returns the population genotype variance of a particular locus. */
  double locusVar(std::size_t loc) const noexcept { return locus_vars_(loc); }

  /** \brief Returns the population allele frequency at a particular locus. */
  double alleleFreq(std::size_t loc) const noexcept {
    return allele_freqs_(loc);
  }

  /** \brief Returns the number of individuals. */
  std::size_t individuals() const noexcept { return paternal_.individuals(); }

  /** \brief Returns the number of loci. */
  std::size_t loci() const noexcept { return paternal_.loci(); }

  /** \brief Returns the number of rows in the buffer. */
  std::size_t rows() const noexcept { return paternal_.rows(); }

  /** \brief Returns the number of words in the buffer. */
  std::size_t words() const noexcept { return paternal_.words(); }

  /** \brief Returns the paternal haplotype buffer. */
  HaplotypeBuffer& paternal() noexcept { return paternal_; }

  /** \brief Returns the maternal haplotype buffer. */
  HaplotypeBuffer& maternal() noexcept { return maternal_; }

  /** \brief Returns the paternal haplotype buffer. */
  const HaplotypeBuffer& paternal() const noexcept { return paternal_; }

  /** \brief Returns the maternal haplotype buffer. */
  const HaplotypeBuffer& maternal() const noexcept { return maternal_; }

  /** \brief Returns the buffer layout. */
  BufferLayout view() const noexcept { return paternal_.layout(); }

  /** \brief Transposes the genotype buffer. */
  void transpose() noexcept;

  /** \brief Computes summary statistics of loci. */
  void computeLocusStats();

  /**
   * \brief Decompresses a portion of a genotype buffer into a double matrix.
   */
  void decompress(
      std::size_t individual_start,
      std::size_t individual_end,
      const std::vector<std::size_t>& loci,
      Eigen::MatrixXd& buffer_out,
      bool centre = false,
      bool scale = false);

 private:
  /** */
  const Eigen::VectorXd locus_mut_;
  const Eigen::VectorXd locus_rec_;
  const Eigen::VectorXd allele_freqs_init_;
  Eigen::VectorXd locus_means_;
  Eigen::VectorXd locus_vars_;
  Eigen::VectorXd allele_freqs_;
  Eigen::MatrixXd genotype_freqs_;
  HaplotypeBuffer paternal_;
  HaplotypeBuffer maternal_;
};

/**
 * \details Transposes both constituent haplotype buffers simultaneously.
 *
 * \see BufferLayout
 * \see HaplotypeBuffer::transpose()
 */
inline void GenotypeBuffer::transpose() noexcept {
  paternal_.transpose();
  maternal_.transpose();
}

/**
 * \details Single-pass computation of locus summary statistics, the results of
 * which are stored in the fields \ref locus_means_, \ref locus_vars_, \ref
 * allele_freqs_, and \ref genotype_freqs_. These are used in subsequent
 * computations for efficient, in-place standardisation of genotypes.
 *
 * \pre Buffer must be in locus-major layout
 */
inline void GenotypeBuffer::computeLocusStats() {
  assert(
      (paternal_.layout() == BufferLayout::LocusMajor &&
       maternal_.layout() == BufferLayout::LocusMajor) &&
      "requires locus-major layout");

  std::size_t n_ind = individuals();
  std::size_t n_loci = loci();

  const std::size_t n_blocks = (n_ind + 63) / 64;

  for (std::size_t loc = 0; loc < n_loci; ++loc) {
    std::size_t hom = 0;
    std::size_t het = 0;
    std::size_t dosage = 0;

    for (std::size_t block = 0; block < n_blocks; ++block) {
      hom +=
          __builtin_popcountll((paternal_(loc, block) & maternal_(loc, block)));
      het +=
          __builtin_popcountll((paternal_(loc, block) ^ maternal_(loc, block)));
      dosage += hom + het;
    }

    /** trim away the bits used to pad the buffer to a multiple of 64 */
    std::uint64_t mask = (n_ind % 64) ? ~((1ULL << (n_ind % 64)) - 1ULL) : 0ULL;
    std::size_t homl = __builtin_popcountll(
        (paternal_(loc, n_blocks - 1) & maternal_(loc, n_blocks - 1)) & mask);
    std::size_t hetl = __builtin_popcountll(
        (paternal_(loc, n_blocks - 1) ^ maternal_(loc, n_blocks - 1)) & mask);

    hom -= homl;
    het -= hetl;
    dosage -= homl + hetl;

    allele_freqs_(loc) =
        static_cast<double>(dosage) / (2.0 * static_cast<double>(n_ind));

    genotype_freqs_(0, loc) =
        static_cast<double>((n_ind - hom - het)) / static_cast<double>(n_ind);
    genotype_freqs_(1, loc) =
        static_cast<double>(het) / static_cast<double>(n_ind);
    genotype_freqs_(2, loc) =
        static_cast<double>(hom) / static_cast<double>(n_ind);

    double mean_loc = static_cast<double>((2.0 * hom) + het) / n_ind;
    double mean_sqloc = static_cast<double>((4.0 * hom) + het) / n_ind;
    double var_loc = mean_sqloc - (mean_loc * mean_loc);

    locus_means_(loc) = mean_loc;
    locus_vars_(loc) = var_loc;
  }
}

/**
 * \details Rows index individuals in the range [\p individual_start, \p
 * individual_end), with columns running over the loci listed in \p loci, in
 * order of appearance. If \p buffer_out is larger than the requested data,
 * the decompressed data are written in the top left block matrix of the
 * destination.
 *
 * \param individual_start Individual index where decompression is started.
 * \param individual_end Individual index where decompression stops, exclusive.
 * \param loci A vector of locus indices which are decompressed
 * \param buffer_out Eigen matrix which decompressed data is written into
 * \param centre If `true`, genotypes in decompressed matrix have zero mean.
 * \param scale If `true`, genotypes in decompressed matrix have unit variance.
 *
 * \pre Buffer must be in locus-major layout.
 * \pre \p individual_start <= \p individual_end must lie in [0, \ref n_ind_].
 * \pre Locus indices must lie in [0, \ref n_loci_).
 * \pre Output buffer must be large enough to house decompressed matrix.
 */
inline void GenotypeBuffer::decompress(
    std::size_t individual_start,
    std::size_t individual_end,
    const std::vector<std::size_t>& loci,
    Eigen::MatrixXd& buffer_out,
    bool centre,
    bool scale) {
  assert(
      view() == BufferLayout::LocusMajor && "locus-major layout is required");
  assert(
      individual_start < individual_end && individual_end <= individuals() &&
      "individual indices are out of bounds or specify an empty interval");
  assert(
      buffer_out.rows() >=
          static_cast<Eigen::Index>(individual_end - individual_start) &&
      buffer_out.cols() >= static_cast<Eigen::Index>(loci.size()) &&
      "output buffer is too small for decompressed data");
  assert(
      std::ranges::all_of(
          loci, [this](std::size_t loc) { return loc < this->loci(); }) &&
      "locus index / indices are out of bounds");

  std::size_t n_loci = loci.size();
  std::size_t start = individual_start / 64;
  std::size_t end = (individual_end + 63) / 64;

  for (std::size_t el = 0; el < n_loci; ++el) {
    double scl = (scale) ? 1.0 / std::sqrt(locus_vars_[loci[el]]) : 1.0;
    double cen = (centre) ? -locus_means_[loci[el]] * scl : 0;

    for (std::size_t word = start; word < end; ++word) {
      std::size_t bit_lo = (word == start) ? (individual_start % 64) : 0;
      std::size_t bit_hi =
          (word == end - 1) ? ((individual_end - 1) % 64) + 1 : 64;
      std::uint64_t paternal = paternal_(loci[el], word);
      std::uint64_t maternal = maternal_(loci[el], word);

      for (std::size_t bit = bit_lo; bit < bit_hi; ++bit) {
        buffer_out((64 * word) + bit - individual_start, el) =
            (scl * (((paternal >> bit) & 1ULL) + ((maternal >> bit) & 1ULL))) +
            cen;
      }
    }
  }
}

}  // namespace knoop
