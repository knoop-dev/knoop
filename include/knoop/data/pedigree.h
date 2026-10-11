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
#include <knoop/data/matching.h>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <format>
#include <ranges>
#include <utility>

namespace knoop {

class Pedigree;

/**
 * Represents the atomic unit of the population, the individual. Each individual
 * has an index which locates them within the data buffers, and a depth which
 * specifies which generation they are in. Moreover, each individual holds a
 * reference to a pedigree which allows traversal of the pedigree.
 */
struct Individual {
  std::size_t index;
  std::size_t depth;
  const Pedigree* pedigree;

  bool operator==(const Individual&) const = default;

  bool isMale() const;

  // these are immediate
  Individual spouse() const;
  Individual sibling() const;
  Individual father() const;
  Individual mother() const;
  Individual son() const;
  Individual daughter() const;

  // these are "generic"
  Individual parent(Sex sex) const;
  Individual child(Sex sex) const;
};

struct PedigreePath {
  std::uint64_t seq;
  std::size_t seq_len;
  std::optional<std::size_t> pivot;
};

class Pedigree {
 public:
  explicit Pedigree(std::size_t max_depth, std::size_t n_ind)
      : max_size_(max_depth), n_ind_(n_ind), n_sex_(n_ind / 2) {}

  void push(const Matching& matching, const Matching& inv_matching);

  std::size_t size() const { return history_.first.size(); }

  std::size_t maxSize() const { return max_size_; }

  std::size_t numInd() const { return n_ind_; }

  std::size_t numSex() const { return n_sex_; }

  bool isMale(const Individual& self) const;

  std::size_t spouseIndex(const Individual& self) const;
  std::size_t fatherIndex(const Individual& self) const;
  std::size_t motherIndex(const Individual& self) const;
  std::size_t siblingIndex(const Individual& self) const;
  Individual at(std::size_t index, std::size_t depth) const;

  static std::vector<Individual> traversal(
      const Individual& init, const PedigreePath& path);

  static std::vector<std::vector<Individual>> getCousins(
      const Individual& self, std::size_t degree = 1);

  std::vector<std::vector<Individual>> getSiblings() const;

  std::vector<std::vector<Individual>> getCousins(std::size_t degree = 1) const;

  static std::vector<std::vector<Individual>> getAncestors(
      const Individual& self, std::size_t degree = 1);

  std::vector<std::vector<Individual>> getAncestors(
      std::size_t degree = 1) const;

  auto getSpousePairs(std::size_t depth = 1) const;

 private:
  std::size_t max_size_;
  std::size_t n_ind_;
  std::size_t n_sex_;
  std::pair<std::deque<Matching>, std::deque<Matching>> history_;

  bool isMaxDepth(std::size_t depth) const {
    return (depth == max_size_) || (this->size() < depth + 2);
  }
};

inline void Pedigree::push(
    const Matching& matching, const Matching& inv_matching) {
  if (size() == max_size_ + 1) {
    history_.first.pop_back();
    history_.second.pop_back();
  }
  history_.first.emplace_front(matching);
  history_.second.emplace_front(inv_matching);
}

inline Individual Pedigree::at(std::size_t index, std::size_t depth) const {
  if (index > n_ind_)
    throw std::invalid_argument(
        std::format("Individual index must be in range [0, {}]", n_ind_ - 1));

  if (depth >= max_size_)
    throw std::invalid_argument(
        std::format(
            "Pedigree::at depth {} exceeds pedigree depth {}",
            depth,
            max_size_));

  return Individual{.index = index, .depth = depth, .pedigree = this};
}

inline bool Pedigree::isMale(const Individual& self) const {
  return self.index < n_sex_;
}

inline std::size_t Pedigree::spouseIndex(const Individual& self) const {
  return isMale(self) ? n_sex_ + history_.first[self.depth][self.index]
                      : history_.second[self.depth][self.index - n_sex_];
}

inline std::size_t Pedigree::fatherIndex(const Individual& self) const {
  if (isMaxDepth(self.depth))
    throw std::runtime_error(
        "Pedigree::fatherIndex: Individual is at maximum depth");

  return isMale(self) ? self.index
                      : history_.second[self.depth + 1][self.index - n_sex_];
}

inline std::size_t Pedigree::motherIndex(const Individual& self) const {
  if (isMaxDepth(self.depth))
    throw std::runtime_error(
        "Pedigree::motherIndex: Individual is at maximum depth");

  return isMale(self) ? n_sex_ + history_.first[self.depth + 1][self.index]
                      : self.index;
}

inline std::size_t Pedigree::siblingIndex(const Individual& self) const {
  if (isMaxDepth(self.depth))
    throw std::runtime_error(
        "Pedigree::siblingIndex: Individual is at maximum depth");

  return isMale(self) ? n_sex_ + history_.first[self.depth + 1][self.index]
                      : history_.second[self.depth + 1][self.index - n_sex_];
}

inline bool Individual::isMale() const { return pedigree->isMale(*this); }

inline Individual Individual::father() const {
  return Individual{
      .index = pedigree->fatherIndex(*this),
      .depth = depth + 1,
      .pedigree = pedigree};
}

inline Individual Individual::mother() const {
  return Individual{
      .index = pedigree->motherIndex(*this),
      .depth = depth + 1,
      .pedigree = pedigree};
}

inline Individual Individual::parent(Sex sex) const {
  return (sex == Sex::Male) ? this->father() : this->mother();
}

inline Individual Individual::sibling() const {
  return Individual{
      .index = pedigree->siblingIndex(*this),
      .depth = depth,
      .pedigree = pedigree};
}

inline Individual Individual::spouse() const {
  return Individual{
      .index = pedigree->spouseIndex(*this),
      .depth = depth,
      .pedigree = pedigree};
}

inline Individual Individual::son() const {
  return Individual{
      .index = isMale() ? index : pedigree->spouseIndex(*this),
      .depth = depth - 1,
      .pedigree = pedigree};
}

inline Individual Individual::daughter() const {
  return Individual{
      .index = isMale() ? pedigree->spouseIndex(*this) : index,
      .depth = depth - 1,
      .pedigree = pedigree};
}

inline Individual Individual::child(Sex sex) const {
  return (sex == Sex::Male) ? this->son() : this->daughter();
}

inline std::vector<Individual> Pedigree::traversal(
    const Individual& init, const PedigreePath& path) {
  std::vector<Individual> result(path.seq_len + 1);
  result[0] = init;

  for (std::size_t l = 0; l < path.seq_len; ++l) {
    Sex sex = bitToSex((path.seq >> l) & 1ULL);
    bool up = l < path.pivot.value_or(path.seq_len);

    // If a path is pivoted — that is, can be decomposed into an ascent sequence
    // followed by a descent sequence — then the "apex" node is irrelevant in
    // the bit sequence seeing as couples are strictly monogamous. Thus, the
    // node preceding the apex node simply steps over to its sibling which
    // always yields a valid cousin-cousin sequence.
    if (path.pivot.has_value() && (l == path.pivot.value()))
      result[l + 1] = result[l].sibling().child(sex);
    else
      result[l + 1] = up ? result[l].parent(sex) : result[l].child(sex);
  }

  return result;
}

inline std::vector<std::vector<Individual>> Pedigree::getCousins(
    const Individual& self, std::size_t degree) {
  if (degree == 0)
    throw std::invalid_argument(
        "Pedigree::find_cousins: degree must be non-zero.");

  std::size_t seq_len = 2 * degree;
  std::vector<std::vector<Individual>> cousins;

  // 2^(2d) bits in total
  for (std::uint64_t seq = 0; seq < (1ULL << seq_len); ++seq) {
    PedigreePath path{.seq = seq, .seq_len = seq_len, .pivot = degree};
    cousins.push_back(traversal(self, path));
  }

  return cousins;
}

inline std::vector<std::vector<Individual>> Pedigree::getSiblings() const {
  std::vector<std::vector<Individual>> siblings(n_ind_);

  for (std::size_t ind = 0; ind < n_ind_; ++ind) {
    Individual self = at(ind, 0);
    Individual sib = self.sibling();
    siblings[ind] = {self, sib};
  }

  return siblings;
}

inline std::vector<std::vector<Individual>> Pedigree::getCousins(
    std::size_t degree) const {
  if (degree == 0)
    throw std::invalid_argument(
        "Pedigree::find_cousins: degree must be non-zero.");

  auto cousins_per_ind = static_cast<std::size_t>(1ULL << (2 * degree));
  std::vector<std::vector<Individual>> cousins(n_ind_ * cousins_per_ind);

  for (std::size_t ind = 0; ind < n_ind_; ++ind) {
    auto ind_cousins = getCousins(at(ind, 0), degree);
    std::ranges::move(ind_cousins, (cousins.begin() + (ind * cousins_per_ind)));
  }

  return cousins;
}

inline std::vector<std::vector<Individual>> Pedigree::getAncestors(
    const Individual& self, std::size_t degree) {
  if (degree == 0)
    throw std::invalid_argument(
        "Pedigree::find_ancestors: degree must be non-zero");

  std::size_t seq_len = degree;
  std::vector<std::vector<Individual>> ancestors;

  for (std::uint64_t seq = 0; seq < (1ULL << seq_len); ++seq) {
    PedigreePath path{.seq = seq, .seq_len = seq_len};
    ancestors.push_back(traversal(self, path));
  }

  return ancestors;
}

inline std::vector<std::vector<Individual>> Pedigree::getAncestors(
    std::size_t degree) const {
  if (degree == 0)
    throw std::invalid_argument(
        "Pedigree::find_ancestors: degree must be non-zero");

  auto ancestors_per_ind = static_cast<std::size_t>(1ULL << degree);
  std::vector<std::vector<Individual>> ancestors(n_ind_ * ancestors_per_ind);

  for (std::size_t ind = 0; ind < n_ind_; ++ind) {
    auto ind_ancestors = getAncestors(at(ind, 0), degree);
    std::ranges::move(
        ind_ancestors, ancestors.begin() + (ancestors_per_ind * ind));
  }

  return ancestors;
}

inline auto Pedigree::getSpousePairs(std::size_t depth) const {
  auto males = std::views::iota(std::size_t{0}, n_sex_) |
               std::views::transform([this, depth](std::size_t idx) {
                 return this->at(idx, depth);
               });

  auto females = males | std::views::transform([](const Individual& ind) {
                   return ind.spouse();
                 });

  return std::views::zip(males, females);
}

}  // namespace knoop
