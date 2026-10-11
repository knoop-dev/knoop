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
#include <knoop/estimate/sample.h>
#include <knoop/estimate/sample_factory.h>
#include <knoop/io/parse.h>
#include <knoop/sample/proband.h>
#include <knoop/sample/sample.h>

#include <filesystem>
#include <numeric>
#include <ranges>
#include <variant>

namespace knoop {

// TODO: this should be an accessor in PhenotypeBuffer
inline double individualPhenotype(
    const State& state,
    const Individual& ind,
    std::size_t pheno_id,
    Component component = Component::Total) {
  Generation gen = (ind.depth == 0) ? Generation::Current : Generation::Parents;
  return state.pheno(gen)(pheno_id, component)(ind.index);
}

// TODO: this should be an accessor in GenotypeBuffer
inline std::uint8_t individualGenotypePLINK(
    const State& state, const Individual& ind, std::size_t loc) {
  Generation gen = (ind.depth == 0) ? Generation::Current : Generation::Parents;
  std::size_t word = ind.index / 64;
  std::size_t bit = ind.index % 64;

  std::uint8_t h0 = (state.geno(gen).paternal()(loc, word) >> bit) & 1;
  std::uint8_t h1 = (state.geno(gen).maternal()(loc, word) >> bit) & 1;

  return (h0 & h1) | ((h0 | h1) << 1);
}

template <ProbandType P>
inline void SampleFor<P>::attachEstimator(const SampleEstimatorSpec& spec) {
  estimators.push_back(buildSampleEstimator<P>(spec));
}

/**
 * @brief Class which performs sampling on the population
 *
 * The Sampler class produces subsamples of the population which can mimic
 * selection biases such as participation and ascertainment. This is done via
 * a probability weighting scheme whereby
 */
class Sampler {
 public:
  template <ProbandType P>
  explicit Sampler(SampleFor<P> sample, const Params& params)
      : self_(std::make_unique<Model<P>>(std::move(sample), params)){};

  void operator()(const State& state) { (*self_)(state); }

 private:
  // type-erased computing interface
  struct Concept {
    virtual ~Concept() = default;
    virtual void draw(const State& state) = 0;
    virtual void estimate(const State& state) = 0;
    virtual void operator()(const State& state) = 0;
  };

  // implementation strategy
  template <ProbandType P>
  struct Model : Concept {
    explicit Model(SampleFor<P> sample, const Params& params)
        : eng(random::Xoshiro256pp::getInstance()),
          name(sample.name),
          n_probands(sample.n_probands),
          sample_dir(params.global.out_dir / name),
          of(sample.of),
          include(sample.include),
          agg(std::move(sample.agg)),
          weighting(std::move(sample.weighting)),
          decompress_genotypes(sample.decompress_genotypes),
          n_probands_total(
              P == ProbandType::Self ? params.global.n_ind
                                     : params.global.n_ind / 2),
          n_sex(params.global.n_ind / 2),
          n_of(
              __builtin_popcountll(static_cast<ProbandEnumType<P>>(sample.of))),
          n_pheno(params.pheno.n_pheno),
          n_loc(params.geno.n_loc),
          n_on(sample.on.has_value() ? sample.on.value().size() : 1),
          on_components(
              sample.on_components.has_value()
                  ? std::move(sample.on_components.value())
                  : std::vector<Component>(n_on, Component::Total)),
          preaggregate(n_of, n_on),
          aggregates(n_probands_total, n_on),
          unif(n_probands_total),
          keys(n_probands_total),
          names(params.pheno.names),
          phenotypes(sample.n_probands * kProbandSize<P>, n_pheno),
          selected(n_probands_total) {
      if (n_probands > n_probands_total)
        throw std::runtime_error(
            "n_probands (" + std::to_string(n_probands) + ") exceeds " +
            "available sampling units (" + std::to_string(n_probands_total) +
            ") for sample '" + name + "'");

      if (sample.on.has_value())
        for (const auto& pheno_name : sample.on.value())
          on_indices.push_back(params.pheno.pheno_ids.at(pheno_name));

      std::filesystem::create_directories(sample_dir);
      for (const auto& estimator : sample.estimators)
        estimators.emplace_back(estimator(params, n_probands, sample_dir));
    };

    // tra-la-la
    boost::random::xoshiro256pp& eng;

    // fields obtained from sampler specification
    std::string name;
    std::size_t n_probands;
    std::filesystem::path sample_dir;
    ProbandMemberEnum<P> of;
    ProbandMemberEnum<P> include;
    AggFunction agg;
    WeightFunction weighting;
    std::vector<std::unique_ptr<SampleEstimatorStrategy<P>>> estimators;
    bool decompress_genotypes;

    // derived dimensions
    std::size_t n_probands_total;
    std::size_t n_sex;
    std::size_t n_of;
    std::size_t n_pheno;
    std::size_t n_loc;
    std::size_t n_on;

    // phenotype indices and components for sampling
    std::vector<std::size_t> on_indices;
    std::vector<Component> on_components;

    // buffers to store sampling keys and data
    Eigen::MatrixXd preaggregate;
    Eigen::MatrixXd aggregates;
    Eigen::VectorXd unif;
    Eigen::VectorXd keys;

    // extracted proband phenotypes
    std::vector<std::string> names;
    Eigen::MatrixXd phenotypes;
    Eigen::MatrixXd genotypes;
    std::vector<std::size_t> selected;

    void fillAggregates(const State& state);

    // TODO: migrate these into io/, should not be responsibility of sampler
    void writeGenotype(
        const State& state,
        const Individual& ind,
        std::size_t loc,
        std::uint8_t& byte,
        std::size_t& bit_pos,
        std::fstream& bed) const;
    void writeBim() const;
    void writeBed(const State& state) const;
    void writeFam(const State& state) const;
    void writePheno(
        const State& state, Component type = Component::Total) const;
    void writePlink(const State& state) const {
      writeBim();
      writeBed(state);
      writeFam(state);
      for (auto comp :
           {Component::Total,
            Component::Genetic,
            Component::Environmental,
            Component::Vertical})
        writePheno(state, comp);
    }

    void draw(const State& state) override;
    void estimate(const State& state) override;
    void operator()(const State& state) override;
  };

  std::unique_ptr<Concept> self_;
};

template <ProbandType P>
inline void Sampler::Model<P>::fillAggregates(const State& state) {
  if (on_indices.empty()) return;

  for (const Proband<P>& proband : getProbands<P>(state.pedigree)) {
    // fill preaggregation buffer
    for (std::size_t on = 0; on < n_on; ++on) {
      std::size_t pheno_id = on_indices[on];
      Component component = on_components[on];
      std::size_t mem_pos = 0;

      for (std::size_t m = 0; m < kProbandSize<P>; ++m) {
        if ((proband.memberData(m).self & of) == ProbandMemberEnum<P>{})
          continue;

        preaggregate(mem_pos, on) =
            individualPhenotype(state, proband.member[m], pheno_id, component);

        ++mem_pos;
      }
    }

    // apply aggregation
    aggregates.row(proband.id.index) = aggregate(preaggregate, agg);
  }
}

template <ProbandType P>
inline void Sampler::Model<P>::writeGenotype(
    const State& state,
    const Individual& ind,
    std::size_t loc,
    std::uint8_t& byte,
    std::size_t& bit_pos,
    std::fstream& bed) const {
  byte |= (individualGenotypePLINK(state, ind, loc) << bit_pos);
  bit_pos += 2;
  if (bit_pos == 8) {
    bed.put(byte);
    byte = 0;
    bit_pos = 0;
  }
}

template <ProbandType P>
inline void Sampler::Model<P>::writeBim() const {
  Table<
      Column<"chrom", std::size_t>,
      Column<"variant_id", std::string>,
      Column<"pos_cm", std::size_t>,
      Column<"bp_coordinate", std::size_t>,
      Column<"alt", std::string>,
      Column<"ref", std::string>>
      bim;

  for (std::size_t loc = 0; loc < n_loc; ++loc)
    bim.insertRow({1, std::format("SNP{}", loc), 0, loc + 1, "A", "G"});

  writeTable(bim, sample_dir / "data.bim", '\t', '\n', false);
}

template <ProbandType P>
inline void Sampler::Model<P>::writeBed(const State& state) const {
  if (state.geno().view() != BufferLayout::LocusMajor)
    throw std::runtime_error(
        "Sampler::Model<P>::writeBED: require locus-major layout");

  // open the file
  std::fstream bed_file(
      sample_dir / "data.bed", std::ios::out | std::ios::binary);

  if (!bed_file.is_open())
    throw std::runtime_error(
        "Could not open BED file stream " + (sample_dir / "data.bed").string());

  // magic numbers
  bed_file.put(0x6c).put(0x1b).put(0x01);

  // retrieve the probands we want
  auto probands = getProbands<P>(state.pedigree);

  for (std::size_t loc = 0; loc < n_loc; ++loc) {
    std::uint8_t byte = 0;
    std::size_t bit_pos = 0;

    for (std::size_t sel : selected | std::views::take(n_probands)) {
      Proband<P> proband = probands[sel];

      for (std::size_t m = 0; m < proband.size(); ++m) {
        if ((proband.memberData(m).self & include) == ProbandMemberEnum<P>{})
          continue;

        writeGenotype(state, proband.member[m], loc, byte, bit_pos, bed_file);
      }
    }

    // flush partial buffer at end of locus
    if (bit_pos != 0) bed_file.put(byte);
  }
}

template <ProbandType P>
inline void Sampler::Model<P>::writeFam(const State& state) const {
  Table<
      Column<"FID", std::string>,
      Column<"IID", std::string>,
      Column<"PID", std::string>,
      Column<"MID", std::string>,
      Column<"SEX", int>,
      Column<"PHENO", double>>
      table_fam;

  auto member_id = [this](const Proband<P>& proband, const Individual& who) {
    for (std::size_t m = 0; m < proband.size(); ++m)
      if (proband.member[m] == who) {
        if ((proband.memberData(m).self & include) == ProbandMemberEnum<P>{})
          return std::string("0");

        return std::format(
            "{}{}", proband.memberData(m).code, proband.member[m].index);
      }
    return std::string("0");
  };

  auto probands = getProbands<P>(state.pedigree);

  for (std::size_t sel : std::views::take(selected, n_probands)) {
    Proband<P> proband = probands[sel];

    for (std::size_t m = 0; m < proband.size(); ++m) {
      if ((proband.memberData(m).self & include) == ProbandMemberEnum<P>{})
        continue;

      const Individual& individual = proband.member[m];

      std::string fam_id = std::format("FAM{}", proband.id.index);
      std::string self_id =
          std::format("{}{}", proband.memberData(m).code, individual.index);

      // check if father / mother are included before writing
      std::string father_id = member_id(proband, individual.father());
      std::string mother_id = member_id(proband, individual.mother());
      int sex = individual.isMale() ? 1 : 2;

      table_fam.insertRow({fam_id, self_id, father_id, mother_id, sex, -9});
    }
  }

  writeTable(table_fam, sample_dir / "data.fam", '\t', '\n', false);
}

template <ProbandType P>
inline void Sampler::Model<P>::writePheno(
    const State& state, Component type) const {
  auto path =
      (type == Component::Total)
          ? sample_dir / "data.pheno"
          : sample_dir / std::format("data.{}.pheno", componentToString(type));

  std::fstream pheno_file(path, std::ios::out);

  if (!pheno_file.is_open())
    throw std::runtime_error(
        "Could not open PHENO file output stream " + path.string());

  std::string header = "FID\tIID";
  for (const std::string& pheno : names) header += "\t" + pheno;
  header += "\n";

  pheno_file << header;

  auto probands = getProbands<P>(state.pedigree);

  for (std::size_t sel : selected | std::views::take(n_probands)) {
    Proband<P> proband = probands[sel];

    for (std::size_t m = 0; m < proband.size(); ++m) {
      if ((proband.memberData(m).self & include) == ProbandMemberEnum<P>{})
        continue;

      const Individual& member = proband.member[m];

      pheno_file << std::format(
          "FAM{}\t{}{}",
          proband.id.index,
          proband.memberData(m).code,
          member.index);

      for (std::size_t pheno = 0; pheno < n_pheno; ++pheno)
        pheno_file << std::format(
            "\t{}", individualPhenotype(state, member, pheno, type));

      pheno_file << "\n";
    }
  }
}

template <ProbandType P>
inline void Sampler::Model<P>::draw(const State& state) {
  // fill the aggregation buffer with phenotypes of the respective members
  fillAggregates(state);

  // generate uniform weights for Efraimidis-Spirakis
  random::Uniform::sample(eng, unif.data(), n_probands_total);

  // run the weighting fn on the aggregation buffer
  weighting(aggregates, keys);
  keys = unif.array().pow(1.0 / keys.array());

  // NOLINTBEGIN(modernize-use-ranges)
  std::iota(selected.begin(), selected.end(), 0);
  // NOLINTEND(modernize-use-ranges)

  std::nth_element(
      selected.begin(),
      selected.begin() + n_probands,
      selected.end(),
      [&](auto a, auto b) { return keys(a) > keys(b); });

  writePlink(state);
}

template <ProbandType P>
inline void Sampler::Model<P>::estimate(const State& state) {
  for (const auto& estimator : estimators) (*estimator)(state.gen, state.rep);
}

template <ProbandType P>
inline void Sampler::Model<P>::operator()(const State& state) {
  std::vector<std::filesystem::path> to_remove;
  for (const auto& entry : std::filesystem::directory_iterator(sample_dir))
    to_remove.push_back(entry.path());
  for (const auto& p : to_remove) std::filesystem::remove(p);
  draw(state);
  estimate(state);
}

template struct Sampler::Model<ProbandType::Self>;
template struct Sampler::Model<ProbandType::Mate>;
template struct Sampler::Model<ProbandType::Family>;

class ComputeSampleEstimates {
 public:
  explicit ComputeSampleEstimates(const Params& params) {
    for (const auto& sample : params.estimate.samples)
      estimators_.emplace_back(
          std::visit(
              [&params](auto&& spec) { return Sampler(spec, params); },
              sample));
  }

  void operator()(const State& state);

 private:
  std::vector<Sampler> estimators_;
};

inline void ComputeSampleEstimates::operator()(const State& state) {
  for (auto& estimator : estimators_) estimator(state);
}

}  // namespace knoop
