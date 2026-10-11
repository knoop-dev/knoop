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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace knoop {

namespace details {

/**
 * \brief Swaps
 */
inline void bitMatrixSwap(
    std::uint64_t matrix[], std::size_t width, std::uint64_t mask) {
  static constexpr std::size_t kWordSize = 64;
  std::size_t inner;
  std::size_t outer;

  for (outer = 0; outer < kWordSize / (width * 2); ++outer) {
    for (inner = 0; inner < width; ++inner) {
      std::uint64_t* x = &matrix[(inner) + (outer * width * 2)];
      std::uint64_t* y = &matrix[(inner + width) + (outer * width * 2)];
      *x = ((*y << width) & mask) ^ *x;
      *y = ((*x & mask) >> width) ^ *y;
      *x = ((*y << width) & mask) ^ *x;
    }
  }
}

/**
 * \brief Transposes a bit matrix.
 */
inline void bitMatrixTranspose(std::uint64_t* matrix) {
  static constexpr std::size_t kSwapWidthInit = 64;
  std::size_t swap_width = kSwapWidthInit;
  auto swap_mask = static_cast<std::uint64_t>(-1);
  while (swap_width != 1) {
    swap_width >>= 1;
    swap_mask = swap_mask ^ (swap_mask >> swap_width);
    bitMatrixSwap(matrix, swap_width, swap_mask);
  }
}

}  // namespace details

/**
 * \brief Layout for bit-packed genome buffers.
 *
 * \details In knoop, haplotypes are stored in either locus or individual major
 * layout. In locus major layout, individuals' haplotypes are bit-packed into
 * 64-bit unsigned integers, such that within a single locus `l`, the `j`th word
 * contains the haplotypes of individuals with indices between `64 * j` and
 * `64 * j + 63`.
 *
 * Individual major layout is the inverse of this layout, with each word
 * containing 64 haplotypes at a time. In both cases, the number of words in the
 * buffer equals the nearest multiple of 64 that exceeds the number of loci or
 * individuals, respectively.
 *
 * \see HaplotypeBuffer
 * \see GenotypeBuffer
 */
enum class BufferLayout : bool {
  /** \brief Rows consist of */
  LocusMajor,

  /** */
  IndividualMajor
};

class HaplotypeBuffer {
 public:
  HaplotypeBuffer(std::size_t n_individuals, std::size_t n_loc)
      : n_individuals_(n_individuals),
        n_loci_(n_loc),
        n_rows_((n_loci_ + 63) & ~static_cast<std::size_t>(63)),
        n_words_((n_individuals_ + 63) / 64),
        buf_(n_rows_ * n_words_) {}

  /** \brief Return the number of individuals. */
  std::size_t individuals() const noexcept { return n_individuals_; }

  /** \brief Return the number of loci. */
  std::size_t loci() const noexcept { return n_loci_; }

  /** \brief Return the number of rows in the buffer. */
  std::size_t rows() const noexcept { return n_rows_; }

  /** \brief Return the number of words in the buffer. */
  std::size_t words() const noexcept { return n_words_; }

  /** \brief Return the buffer layout. */
  BufferLayout layout() const noexcept { return layout_; }

  /**
   * \brief Access word \p word in row \p row.
   *
   * \details In individual major layout, row \row contains the haplotypes for
   * loci
   *
   * \param row Row index.
   * \param word Word index.
   *
   * \return Word at index \word in row \row.
   */
  std::uint64_t& operator()(std::size_t row, std::size_t word) noexcept {
    assert(row < n_rows_ && "row index out of bounds");
    assert(word < n_words_ && "word index out of bounds");
    return buf_[(row * n_words_) + word];
  }

  /**
   * \brief Access the word \word in row \row.
   *
   * \param row
   * \param word
   *
   * \return Word at index \word in row \row.
   */
  const std::uint64_t& operator()(
      std::size_t row, std::size_t word) const noexcept {
    assert(row < n_rows_ && "row index out of bounds");
    assert(word < n_words_ && "word index out of bounds");
    return buf_[(row * n_words_) + word];
  }

  /** \brief Returns a pointer to a */
  std::uint64_t* rowptr(std::size_t row) noexcept {
    assert(row < n_rows_ && "row index out of bounds");
    return &buf_[row * n_words_];
  }

  /** \brief */
  const std::uint64_t* rowptr(std::size_t row) const noexcept {
    assert(row < n_rows_ && "row index out of bounds");
    return &buf_[row * n_words_];
  }

  /** \brief Transposes the layout of the buffer. */
  void transpose() noexcept;

 private:
  const std::size_t n_individuals_;
  const std::size_t n_loci_;
  std::size_t n_rows_;
  std::size_t n_words_{};
  std::vector<std::uint64_t> buf_;
  BufferLayout layout_{};
};

/**
 * \details Transposes the buffer
 */
inline void HaplotypeBuffer::transpose() noexcept {
  static constexpr std::size_t kWordSize = 64;

  // pre-transpose
  const std::size_t src_rows = n_rows_;
  const std::size_t src_words = n_words_;

  // post-transpose
  const std::size_t dst_rows =
      (layout_ == BufferLayout::LocusMajor)
          ? ((n_individuals_ + kWordSize - 1) & ~(kWordSize - 1))
          : ((n_loci_ + kWordSize - 1) & ~(kWordSize - 1));

  const std::size_t dst_cols_w =
      (layout_ == BufferLayout::LocusMajor)
          ? ((n_loci_ + kWordSize - 1) / kWordSize)
          : ((n_individuals_ + kWordSize - 1) / kWordSize);

  // allocate destination buffer
  std::vector<std::uint64_t> out(dst_rows * dst_cols_w, 0);

  // src_rows is multiple of 64 by construction
  const std::size_t src_n_rows = src_rows / kWordSize;
  const std::size_t src_n_cols = src_words;

#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t ct = 0; ct < src_n_rows; ++ct) {
    for (std::size_t cb = 0; cb < src_n_cols; ++cb) {
      // NOLINTBEGIN(modernize-avoid-c-arrays)
      std::uint64_t tile[kWordSize];
      // NOLINTEND(modernize-avoid-c-arrays)

      for (std::size_t r = 0; r < kWordSize; ++r)
        tile[r] = (*this)((ct * kWordSize) + r, cb);

      details::bitMatrixTranspose(tile);

      for (std::size_t r = 0; r < kWordSize; ++r) {
        const std::size_t dst_row = (cb * kWordSize) + r;
        const std::size_t dst_col = ct;
        out[(dst_row * dst_cols_w) + dst_col] = tile[r];
      }
    }
  }

  buf_.swap(out);
  n_rows_ = dst_rows;
  n_words_ = dst_cols_w;
  layout_ = (layout_ == BufferLayout::LocusMajor)
                ? BufferLayout::IndividualMajor
                : BufferLayout::LocusMajor;
}

}  // namespace knoop
