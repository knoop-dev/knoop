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

#include <omp.h>

#include <Eigen/Dense>
#include <array>
#include <boost/random/geometric_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/xoshiro.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <vector>

namespace knoop::random {

/**
 * thread-local singleton rng engine wrapping Boost's Xoshiro256++ algorithm
 */
struct Xoshiro256pp {
  static boost::random::xoshiro256pp& getInstance() {
    thread_local static boost::random::xoshiro256pp gen;
    return gen;
  }

  static void setSeed(
      boost::random::xoshiro256pp& engine,
      std::uint64_t seed,
      std::size_t n_skip = 0) {
    boost::random::splitmix64 mix(seed);
    mix.discard(n_skip);
    engine.seed(mix());

    for (int thr = 0; thr < omp_get_thread_num(); ++thr) engine.jump();
  }

  static std::uint64_t setSeed(
      std::optional<std::uint64_t> seed = std::nullopt) {
    std::uint64_t used = seed.value_or(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    getInstance().seed(used);
    return used;
  }

  static std::uint64_t nextUnsignedInt() { return getInstance().next(); }

  static double nextDouble() noexcept {
    return ((nextUnsignedInt() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
};

inline std::uint16_t probThreshold(double p) {
  if (p <= 0.0) return 0;
  if (p >= 1.0 - 0x1p-17) return std::numeric_limits<std::uint16_t>::max();

  constexpr double kScale = 0x1p16;
  constexpr double kMin = 0x1p-16;

  if (p <= kMin)
    throw std::range_error(
        std::format(
            "random::probThreshold: Specified probability {} falls below "
            "granularity of RNG mechanism (threshold: 2^-16, ~1.53e-5)",
            p));

  return static_cast<std::uint16_t>(std::lround(p * kScale));
}

inline std::vector<std::uint16_t> probsToThreshold(
    const std::vector<double>& ps) {
  std::vector<std::uint16_t> thr(ps.size());

  for (std::size_t el = 0; el < ps.size(); ++el)
    thr[el] = probThreshold(ps[el]);

  return thr;
}

inline std::vector<std::uint16_t> probsToThreshold(const Eigen::VectorXd& ps) {
  std::vector<std::uint16_t> thr(ps.size());

  for (std::size_t el = 0; el < ps.size(); ++el)
    thr[el] = probThreshold(ps[el]);

  return thr;
}

inline void probsToThreshold(const Eigen::VectorXd& ps, std::uint16_t* out) {
  for (std::size_t el = 0; el < ps.size(); ++el)
    out[el] = probThreshold(ps[el]);
}

inline bool coinflip(boost::random::xoshiro256pp& engine) {
  return (engine.next() & 1ULL) != 0U;
}

template <std::size_t N>
inline std::array<bool, N> coinflips(boost::random::xoshiro256pp& engine) {
  std::array<bool, N> result;
  std::uint64_t random = 0;

  for (std::size_t el = 0; el < N; ++el) {
    if (el % 64 == 0) random = engine.next();
    result[el] = (random & 1ULL) != 0ULL;
    random >>= 1;
  }

  return result;
}

inline std::uint64_t bernoulliWord(
    boost::random::xoshiro256pp& engine,
    std::uint16_t threshold,
    std::size_t n_valid_bits = 64) noexcept {
  constexpr std::size_t kWordLength = 64;
  constexpr std::size_t kElementWidth = 16;
  constexpr std::size_t kNumPerDraw = 64 / kElementWidth;

  std::uint64_t word = 0;
  std::uint64_t rand_int = 0;

  // indirection per call is far too expensive, retrieve the instance
  for (std::size_t el = 0; el < kWordLength; ++el) {
    if (el % kNumPerDraw == 0) rand_int = engine.next();

    /**
     * retrieve the leftmost bits that corresponds to the 16-bit integer
     * sequence compared against the threshold used to set the individual's bit.
     * once this is read in, shift for next
     */
    std::uint16_t rand_val = rand_int >> (kWordLength - kElementWidth);
    rand_int <<= kElementWidth;

    // insert the thresholded result into the correct bit pos
    word |= std::uint64_t{rand_val < threshold} << el;
  }

  return word & (~0ULL >> (kWordLength - n_valid_bits));
}

inline std::uint64_t bernoulliWord(
    boost::random::xoshiro256pp& engine,
    const std::uint16_t* thresholds,
    std::size_t n_valid_bits = 64) noexcept {
  constexpr std::size_t kWordLength = 64;
  constexpr std::size_t kElementWidth = 16;
  constexpr std::size_t kNumPerDraw = 64 / kElementWidth;

  std::uint64_t word = 0;
  std::uint64_t rand_int = 0;

  for (std::size_t el = 0; el < kWordLength; ++el) {
    if (el % kNumPerDraw == 0) rand_int = engine.next();

    std::size_t rand_val = rand_int >> (kWordLength - kElementWidth);
    rand_int <<= kElementWidth;

    word |= std::uint64_t{rand_val < thresholds[el]} << el;
  }

  return word & (~0ULL >> (kWordLength - n_valid_bits));
}

class Normal {
 public:
  static double sample(
      boost::random::xoshiro256pp& engine,
      double location = 0,
      double scale = 1) {
    return location + (scale * dist_(engine));
  }

  static Eigen::VectorXd sample(
      boost::random::xoshiro256pp& engine,
      std::size_t n,
      double location = 0,
      double scale = 1) {
    Eigen::VectorXd samples(n);
    for (std::size_t el = 0; el < n; ++el)
      samples[el] = sample(engine, location, scale);
    return samples;
  }

  static void sample(
      boost::random::xoshiro256pp& engine,
      double* buf,
      std::size_t n,
      double location = 0,
      double scale = 1) {
    for (std::size_t el = 0; el < n; ++el)
      buf[el] = sample(engine, location, scale);
  }

 private:
  inline static thread_local boost::random::normal_distribution<double> dist_{};

  static double checkScale(double scale) {
    if (scale <= 0)
      throw std::invalid_argument(
          "random::Normal: scale parameter must be positive");
    return scale;
  }
};

class Uniform {
 public:
  static double sample(
      boost::random::xoshiro256pp& engine, double lo = 0, double hi = 1) {
    return lo + ((hi - lo) * dist_(engine));
  }

  static Eigen::VectorXd sample(
      boost::random::xoshiro256pp& engine,
      std::size_t n,
      double lo = 0,
      double hi = 1) {
    Eigen::VectorXd samples(n);
    for (std::size_t el = 0; el < n; ++el) samples[el] = sample(engine, lo, hi);
    return samples;
  }

  static void sample(
      boost::random::xoshiro256pp& engine,
      double* buf,
      std::size_t n,
      double lo = 0,
      double hi = 1) {
    for (std::size_t el = 0; el < n; ++el) buf[el] = sample(engine, lo, hi);
  }

 private:
  inline static thread_local boost::random::uniform_01<double> dist_{};
};

class UniformInt {
 public:
  static std::size_t sample(
      boost::random::xoshiro256pp& engine, std::size_t lo, std::size_t hi) {
    if (hi < lo) throw std::runtime_error("[hi, lo) must be non-empty");
    if (hi == lo) return lo;

    const std::size_t thresh = UINT64_MAX - (UINT64_MAX % (hi - lo));
    std::size_t x;

    do {
      x = engine.next();
    } while (x >= thresh);

    return (x % (hi - lo)) + lo;
  }

  static std::vector<std::size_t> sample(
      boost::random::xoshiro256pp& engine,
      std::size_t n,
      std::size_t lo,
      std::size_t hi) {
    std::vector<std::size_t> samples(n);
    for (std::size_t el = 0; el < n; ++el) samples[el] = sample(engine, lo, hi);
    return samples;
  }

  static void sample(
      boost::random::xoshiro256pp& engine,
      std::size_t* buf,
      std::size_t n,
      std::size_t lo,
      std::size_t hi) {
    for (std::size_t el = 0; el < n; ++el) buf[el] = sample(engine, lo, hi);
  }
};

/**
 * stateful on parametrisation on probability family, so this can't be static
 * like the unstateful classes
 */
class Geometric {
 public:
  explicit Geometric(double p) : dist_(checkP(p)) {}

  double sample(boost::random::xoshiro256pp& engine) { return dist_(engine); }

  Eigen::VectorXd sample(boost::random::xoshiro256pp& engine, std::size_t n) {
    Eigen::VectorXd samples(n);
    for (std::size_t el = 0; el < n; ++el) samples[el] = dist_(engine);
    return samples;
  }

  void sample(boost::random::xoshiro256pp& engin, double* buf, std::size_t n) {
    for (std::size_t el = 0; el < n; ++el) buf[el] = dist_(engin);
  }

 private:
  boost::random::geometric_distribution<std::uint64_t, double> dist_;

  static double checkP(double p) {
    if (p <= 0.0 || p >= 1.0)
      throw std::invalid_argument(
          std::format("random::Geometric: p must lie in (0, 1), got {}", p));
    return p;
  }
};

}  // namespace knoop::random
