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
#include <knoop/core/utils.h>
#include <knoop/data/pedigree.h>
#include <knoop/sample/selection.h>

#include <boost/algorithm/string/case_conv.hpp>
#include <string_view>

namespace knoop {

/**
 * A proband is a collection of individuals related either through familial
 * relations or through spouses (in-laws) that represents the unit of sampling
 * for a given sample. In other words, if we imagine a proband consists of the
 * three members {son, father, mother}, then if the proband is sampled, then
 * all constituents of the proband (son, father, mother) enter the sample.
 *
 * knoop currently supports three proband types:
 *   - Proband::Self: Singleton sets of individuals in the current generation.
 *   - Proband::Family: A sextuple consisting of a nuclear-family plus the
 *       spouses of the children. Spans two generations.
 *   - Proband::Mate: Two nuclear families plus spouses, linked through a single
 *       spouse pair.
 *
 * Currently, intergenerational sampling is unsupported for the Proband::Self
 * type. This is planned for future release.
 *
 * The constituents of the proband type are defined through the enums Self,
 * Family and Mate, with members represented through unsigned integer types.
 * This allows representation of subsets of the proband through the following
 * binary operations:
 *
 *   Family::Son | Family::Father                             (father-son pairs)
 *   HusbandFather | HusbandMother | HusbandSister              (wife's in-laws)
 *   proband_member & Family::Males                              (subset checks)
 */
enum class ProbandType { Self, Family, Mate };

inline std::string probandToString(ProbandType p) {
  if (p == ProbandType::Self) return "self";
  if (p == ProbandType::Family) return "family";
  return "mate";
}

enum class Self : uint8_t { Unknown = 0b0, Self = 0b1 };

enum class Family : uint8_t {
  Unknown = 0b000000,
  Father = 0b000001,
  Mother = 0b000010,
  Son = 0b000100,
  Daughter = 0b001000,
  SonWife = 0b010000,
  DaughterHusband = 0b100000,
  All = 0b111111,
  Parents = Father | Mother,
  Siblings = Son | Daughter,
  Males = Father | Son,
  Females = Mother | Daughter
};

enum class Mate : uint16_t {
  Unknown = 0b0000000000,
  Husband = 0b0000000001,
  Wife = 0b0000000010,
  HusbandFather = 0b0000000100,
  HusbandMother = 0b0000001000,
  HusbandSister = 0b0000010000,
  HusbandBrotherInLaw = 0b0000100000,
  WifeFather = 0b0001000000,
  WifeMother = 0b0010000000,
  WifeBrother = 0b0100000000,
  WifeSisterInLaw = 0b1000000000,
  All = 0b1111111111,
  Couple = Husband | Wife,
  WifeSiblings = Wife | WifeBrother,
  HusbandSiblings = Husband | HusbandSister,
  HusbandInLaws = WifeFather | WifeMother,
  WifeInLaws = HusbandFather | HusbandMother,
  Parents = HusbandInLaws | WifeInLaws,
  HusbandFamily = Husband | WifeInLaws,
  WifeFamily = Wife | HusbandInLaws,
  Males = Husband | HusbandFather | WifeFather,
  Females = Wife | HusbandMother | WifeMother
};

/**
 * Each proband is associated with an ID, which can be interpreted as follows:
 *   - Proband::Self: The index of the individual in the range [0,...,<n_ind>).
 *   - Proband::Family: Index of the father in the range [0,...,<n_ind>/2).
 *   - Proband::Mate: Index of the husband in the range [0,...,<n_ind>/2).
 */
template <ProbandType P>
struct ProbandId {
  std::size_t index;

  // Get all proband identifiers for the given proband type
  static auto all(std::size_t n_ind) {
    return std::views::iota(std::size_t{0}, getProbandRange(n_ind)) |
           std::views::transform([](std::size_t id) { return ProbandId(id); });
  }

 private:
  // creating a ProbandId is disallowed, access is only allowed through
  // all() public function which returns all valid ProbandIds simultaneously
  explicit ProbandId(std::size_t index_) : index(index_) {}

  static std::size_t getProbandRange(std::size_t n_ind) {
    return (P == ProbandType::Self) ? n_ind : n_ind / 2;
  }
};

// Wraps ProbandId<P>::all – retrieves all proband identifiers for a given type
template <ProbandType P>
inline auto getProbandIds(const Pedigree& pedigree) {
  return ProbandId<P>::all(pedigree.numInd());
}

template <typename Enum>
struct Member {
  Enum self;
  Individual (*relation)(const Individual& via);
  std::string_view name;
  std::string_view code;
};

/**
 * Stores compile-time data on sample structure and defaults for each proband
 * type
 */
template <ProbandType P>
struct Data;

template <>
struct Data<ProbandType::Self> {
  using MemberEnum = Self;

  static constexpr MemberEnum kOfDefault = Self::Self;
  static constexpr MemberEnum kIncludeDefault = Self::Self;
  static constexpr AggFunction kAggDefault = AggFunction::Identity;

  static constexpr Member<MemberEnum> kRoot = {
      .self = Self::Self,
      .relation = [](const Individual& via) { return via; },
      .name = "self",
      .code = "IND"};

  static constexpr std::size_t kRootDepth = 0;

  static constexpr std::size_t kNumDerive = 0;
  static constexpr std::array<Member<MemberEnum>, kNumDerive> kDerive = {{}};
};

template <>
struct Data<ProbandType::Family> {
  using MemberEnum = Family;

  static constexpr MemberEnum kOfDefault = Family::All;
  static constexpr MemberEnum kIncludeDefault = Family::All;
  static constexpr AggFunction kAggDefault = AggFunction::Mean;

  static constexpr Member<MemberEnum> kRoot = {
      .self = Family::Father,
      .relation = [](const Individual& root) { return root; },
      .name = "father",
      .code = "FAT"};

  static constexpr std::size_t kRootDepth = 1;

  static constexpr std::size_t kNumDerive = 5;
  static constexpr std::array<Member<MemberEnum>, kNumDerive> kDerive = {{
      {.self = Family::Mother,
       .relation = [](const Individual& root) { return root.spouse(); },
       .name = "mother",
       .code = "MOT"},

      {.self = Family::Son,
       .relation = [](const Individual& root) { return root.son(); },
       .name = "son",
       .code = "SON"},

      {.self = Family::SonWife,
       .relation = [](const Individual& root) { return root.son().spouse(); },
       .name = "son-wife",
       .code = "SOW"},

      {.self = Family::Daughter,
       .relation = [](const Individual& root) { return root.daughter(); },
       .name = "daughter",
       .code = "DAU"},

      {.self = Family::DaughterHusband,
       .relation =
           [](const Individual& root) { return root.daughter().spouse(); },
       .name = "daughter-husband",
       .code = "DAH"},
  }};
};

template <>
struct Data<ProbandType::Mate> {
  using MemberEnum = Mate;

  static constexpr MemberEnum kOfDefault = Mate::Couple;
  static constexpr MemberEnum kIncludeDefault = Mate::All;
  static constexpr AggFunction kAggDefault = AggFunction::Mean;

  static constexpr Member<MemberEnum> kRoot = {
      .self = Mate::Husband,
      .relation = [](const Individual& root) { return root; },
      .name = "husband",
      .code = "hus"};

  static constexpr std::size_t kRootDepth = 0;

  static constexpr std::size_t kNumDerive = 9;
  static constexpr std::array<Member<MemberEnum>, kNumDerive> kDerive = {{
      {.self = Mate::HusbandFather,
       .relation = [](const Individual& root) { return root.father(); },
       .name = "husband-father",
       .code = "HUF"},

      {.self = Mate::HusbandMother,
       .relation = [](const Individual& root) { return root.mother(); },
       .name = "husband-mother",
       .code = "HUM"},

      {.self = Mate::HusbandSister,
       .relation = [](const Individual& root) { return root.sibling(); },
       .name = "husband-sister",
       .code = "HSS"},

      {.self = Mate::HusbandBrotherInLaw,
       .relation =
           [](const Individual& root) { return root.sibling().spouse(); },
       .name = "husband-brother-in-law",
       .code = "HBL"},

      {.self = Mate::Wife,
       .relation = [](const Individual& root) { return root.spouse(); },
       .name = "wife",
       .code = "WIF"},

      {.self = Mate::WifeFather,
       .relation =
           [](const Individual& root) { return root.spouse().father(); },
       .name = "wife-father",
       .code = "WFF"},

      {.self = Mate::WifeMother,
       .relation =
           [](const Individual& root) { return root.spouse().mother(); },
       .name = "wife-mother",
       .code = "WFM"},

      {.self = Mate::WifeBrother,
       .relation =
           [](const Individual& root) { return root.spouse().sibling(); },
       .name = "wife-brother",
       .code = "WFB"},

      {.self = Mate::WifeSisterInLaw,
       .relation =
           [](const Individual& root) {
             return root.spouse().sibling().spouse();
           },
       .name = "wife-sister-in-law",
       .code = "WSL"},
  }};
};

// template wrapper for convenience
template <ProbandType P>
using ProbandMemberEnum = Data<P>::MemberEnum;

template <ProbandType P>
using ProbandEnumType = std::underlying_type_t<ProbandMemberEnum<P>>;

template <ProbandType P>
static constexpr std::size_t kProbandSize = Data<P>::kNumDerive + 1;

template <ProbandType P>
static constexpr ProbandMemberEnum<P> kProbandOfDefault = Data<P>::kOfDefault;

template <ProbandType P>
static constexpr ProbandMemberEnum<P> kProbandIncludeDefault =
    Data<P>::kIncludeDefault;

template <ProbandType P>
static constexpr AggFunction kProbandAggDefault = Data<P>::kAggDefault;

/**
 * An run-time instantiation of a proband, has an associated Id and is filled
 * with its constituent members, represented as Individual types
 */
template <ProbandType P>
struct Proband {
  ProbandId<P> id;
  std::array<Individual, kProbandSize<P>> member;

  static constexpr std::size_t size() { return kProbandSize<P>; }

  static const Member<typename Data<P>::MemberEnum>& memberData(
      std::size_t mem) {
    return (mem == 0) ? Data<P>::kRoot : Data<P>::kDerive[mem - 1];
  }
};

/**
 * Retrieve the proband that corresponds to this supplied ProbandId
 */
template <ProbandType P>
Proband<P> getProband(ProbandId<P> id, const Pedigree& pedigree) {
  std::array<Individual, kProbandSize<P>> member;

  member[0] = pedigree.at(id.index, Data<P>::kRootDepth);
  for (std::size_t mem = 1; mem < kProbandSize<P>; ++mem)
    member[mem] = Data<P>::kDerive[mem - 1].relation(member[0]);

  return Proband<P>{.id = id, .member = member};
}

/**
 * Retrieve all probands and pass them as a transform view
 */
template <ProbandType P>
auto getProbands(const Pedigree& pedigree) {
  return getProbandIds<P>(pedigree) |
         std::views::transform([&](const ProbandId<P>& proband_id) {
           return getProband(proband_id, pedigree);
         });
}

// wrap operations on proband enums

inline Self operator|(Self a, Self b) {
  return static_cast<Self>(
      static_cast<std::underlying_type_t<Self>>(a) |
      static_cast<std::underlying_type_t<Self>>(b));
}

inline Self operator&(Self a, Self b) {
  return static_cast<Self>(
      static_cast<std::underlying_type_t<Self>>(a) &
      static_cast<std::underlying_type_t<Self>>(b));
}

inline Mate operator|(Mate a, Mate b) {
  return static_cast<Mate>(
      static_cast<std::underlying_type_t<Mate>>(a) |
      static_cast<std::underlying_type_t<Mate>>(b));
}

inline Mate operator&(Mate a, Mate b) {
  return static_cast<Mate>(
      static_cast<std::underlying_type_t<Mate>>(a) &
      static_cast<std::underlying_type_t<Mate>>(b));
}

inline Family operator|(Family a, Family b) {
  return static_cast<Family>(
      static_cast<std::underlying_type_t<Family>>(a) |
      static_cast<std::underlying_type_t<Family>>(b));
}

inline Family operator&(Family a, Family b) {
  return static_cast<Family>(
      static_cast<std::underlying_type_t<Family>>(a) &
      static_cast<std::underlying_type_t<Family>>(b));
}

// string interface

template <ProbandType P>
inline ProbandMemberEnum<P> probandEnumFromString(const std::string& s) {
  if (Data<P>::kRoot.name == s) return Data<P>::kRoot.self;
  for (const auto& m : Data<P>::kDerive)
    if (m.name == s) return m.self;

  if constexpr (P == ProbandType::Family) {
    if (s == "all") return Family::All;
    if (s == "parents") return Family::Parents;
    if (s == "siblings") return Family::Siblings;
    if (s == "males") return Family::Males;
    if (s == "females") return Family::Females;
  } else if constexpr (P == ProbandType::Mate) {
    if (s == "all") return Mate::All;
    if (s == "couple") return Mate::Couple;
    if (s == "wife-siblings") return Mate::WifeSiblings;
    if (s == "husband-siblings") return Mate::HusbandSiblings;
    if (s == "husband-in-laws") return Mate::HusbandInLaws;
    if (s == "wife-in-laws") return Mate::WifeInLaws;
    if (s == "parents") return Mate::Parents;
    if (s == "husband-family") return Mate::HusbandFamily;
    if (s == "wife-family") return Mate::WifeFamily;
    if (s == "males") return Mate::Males;
    if (s == "females") return Mate::Females;
  }

  throw std::invalid_argument(
      std::format("Unknown proband member name '{}'", s));
}

}  // namespace knoop
