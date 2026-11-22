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
 * You should have received a copy of the GNU General Public Licensegenome.cc
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <knoop/core/error.h>
#include <omp.h>

#include <Eigen/Dense>
#include <array>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/geometric_distribution.hpp>
#include <boost/random/laplace_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/student_t_distribution.hpp>
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
    throw KnoopRangeError(
        "Specified probability {} falls below "
        "granularity of RNG mechanism (threshold: 2^-16, ~1.53e-5)",
        p);

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

/**
 * \brief Generic probability distribution class for random number generation
 *
 * \details Probability distributions in knoop rely on Boost's random library,
 * specifically the Xoshiro256pp engine. This class is somewhat nuanced in
 * implementation as we want to allow static calls
 */
template <typename Derived>
class Distribution {
 public:
  /** \brief Draw a single random variate */
  template <typename... DistParams>
  static double sample(
      boost::random::xoshiro256pp& engine, DistParams&&... dist_args);

  /** \brief Fill a container of doubles with random variate draws */
  template <typename... DistParams>
  static void sample(
      boost::random::xoshiro256pp& engine,
      double* buf,
      std::size_t n,
      DistParams&&... params);

  /** \brief Draw a vector of randomvariates */
  template <typename... DistParams>
  static Eigen::VectorXd sample(
      boost::random::xoshiro256pp& engine,
      std::size_t n,
      DistParams&&... params);

 private:
  friend Derived;
  Distribution() = default;
};

template <typename Derived>
template <typename... DistParams>
inline double Distribution<Derived>::sample(
    boost::random::xoshiro256pp& engine, DistParams&&... params) {
  return Derived::draw(engine, std::forward<DistParams>(params)...);
}

template <typename Derived>
template <typename... DistParams>
inline void Distribution<Derived>::sample(
    boost::random::xoshiro256pp& engine,
    double* buf,
    std::size_t n,
    DistParams&&... params) {
  for (std::size_t el = 0; el < n; ++el)
    buf[el] = Derived::draw(engine, std::forward<DistParams>(params)...);
}

template <typename Derived>
template <typename... DistParams>
inline Eigen::VectorXd Distribution<Derived>::sample(
    boost::random::xoshiro256pp& engine,
    std::size_t n,
    DistParams&&... params) {
  Eigen::VectorXd result(n);
  sample(engine, result.data(), n, std::forward<DistParams>(params)...);
  return result;
}

class Uniform : public Distribution<Uniform> {
 public:
  using Distribution<Uniform>::sample;

  explicit Uniform(double lo = 0, double hi = 1) : lo_(lo), hi_(hi) {}

  static double draw(
      boost::random::xoshiro256pp& engine, double lo = 0, double hi = 1) {
    thread_local static boost::random::uniform_01<double> unit;
    return lo + ((hi - lo) * unit(engine));
  }

  Eigen::VectorXd operator()(
      boost::random::xoshiro256pp& engine, std::size_t n) const {
    return Uniform::sample(engine, n, lo_, hi_);
  }

  Eigen::VectorXd operator()(std::size_t n) const {
    return Uniform::sample(Xoshiro256pp::getInstance(), n, lo_, hi_);
  }

 private:
  double lo_;
  double hi_;
};

class UniformInt {
 public:
  static std::size_t sample(
      boost::random::xoshiro256pp& engine, std::size_t lo, std::size_t hi) {
    std::size_t range = hi - lo;
    std::uint64_t thresh = UINT64_MAX - (UINT64_MAX % range);
    std::uint64_t x;
    do {
      x = engine.next();
    } while (x >= thresh);
    return (x % range) + lo;
  }
};

class Rademacher : public Distribution<Rademacher> {
 public:
  using Distribution<Rademacher>::sample;

  Rademacher() : Distribution() {}

  static double draw(boost::random::xoshiro256pp& engine) {
    thread_local static boost::random::uniform_01<double> unit;
    return engine.next() & 1ULL ? 1.0 : -1.0;
  }

  Eigen::VectorXd operator()(
      boost::random::xoshiro256pp engine, std::size_t n) const {
    return Rademacher::sample(engine, n);
  }

  Eigen::VectorXd operator()(std::size_t n) const {
    return Rademacher::sample(Xoshiro256pp::getInstance(), n);
  }
};

class Normal : public Distribution<Normal> {
 public:
  using Distribution<Normal>::sample;

  explicit Normal(double location = 0, double scale = 1)
      : location_(location), scale_(scale) {}

  static double draw(
      boost::random::xoshiro256pp& engine,
      double location = 0,
      double scale = 1) {
    thread_local static boost::random::normal_distribution<> dist(0, 1);
    return location + (std::sqrt(scale) * dist(engine));
  }

  Eigen::VectorXd operator()(
      boost::random::xoshiro256pp& engine, std::size_t n) const {
    return Normal::sample(engine, n, location_, scale_);
  }

  Eigen::VectorXd operator()(std::size_t n) const {
    return Normal::sample(Xoshiro256pp::getInstance(), n, location_, scale_);
  }

 private:
  double location_;
  double scale_;
};

class Beta : public Distribution<Beta> {
 public:
  using Distribution<Beta>::sample;

  explicit Beta(double shape1 = 4, double shape2 = 4)
      : shape1_(shape1), shape2_(shape2) {}

  static double draw(
      boost::random::xoshiro256pp& engine,
      double shape1 = 4,
      double shape2 = 4) {
    thread_local static boost::random::beta_distribution<> dist(4, 4);
    thread_local static double dist_shape1 = 4;
    thread_local static double dist_shape2 = 4;

    if (shape1 != dist_shape1 || shape2 != dist_shape2) {
      dist.param(decltype(dist)::param_type(shape1, shape2));
      dist_shape1 = shape1;
      dist_shape2 = shape2;
    }

    return dist(engine);
  }

  Eigen::VectorXd operator()(
      boost::random::xoshiro256pp& engine, std::size_t n) const {
    return Beta::sample(engine, n, shape1_, shape2_);
  }

  Eigen::VectorXd operator()(std::size_t n) const {
    return Beta::sample(Xoshiro256pp::getInstance(), n, shape1_, shape2_);
  }

 private:
  double shape1_;
  double shape2_;
};

class Gamma : public Distribution<Gamma> {
 public:
  using Distribution<Gamma>::sample;

  explicit Gamma(double shape = 1, double scale = 1)
      : shape_(shape), scale_(scale) {}

  static double draw(
      boost::random::xoshiro256pp& engine, double shape = 1, double scale = 1) {
    thread_local static boost::random::gamma_distribution<> dist(1, 1);
    thread_local static double dist_shape = 1;
    thread_local static double dist_scale = 1;

    if (shape != dist_shape || scale != dist_scale) {
      dist.param(decltype(dist)::param_type(shape, scale));
      dist_shape = shape;
      dist_scale = scale;
    }

    return dist(engine);
  }

  Eigen::VectorXd operator()(
      boost::random::xoshiro256pp& engine, std::size_t n) const {
    return Gamma::sample(engine, n, shape_, scale_);
  }

  Eigen::VectorXd operator()(std::size_t n) const {
    return Gamma::sample(Xoshiro256pp::getInstance(), n, shape_, scale_);
  }

 private:
  double shape_;
  double scale_;
};

class Laplace : public Distribution<Laplace> {
 public:
  using Distribution<Laplace>::sample;

  explicit Laplace(double location = 0, double scale = 1)
      : location_(location), scale_(scale) {}

  static double draw(
      boost::random::xoshiro256pp& engine,
      double location = 0,
      double scale = 1) {
    thread_local static boost::random::laplace_distribution<> dist;
    return location + (scale * dist(engine));
  }

  Eigen::VectorXd operator()(
      boost::random::xoshiro256pp& engine, std::size_t n) const {
    return Laplace::sample(engine, n, location_, scale_);
  }

  Eigen::VectorXd operator()(std::size_t n) const {
    return Laplace::sample(Xoshiro256pp::getInstance(), n, location_, scale_);
  }

 private:
  double location_;
  double scale_;
};

class Cauchy : public Distribution<Cauchy> {
 public:
  using Distribution<Cauchy>::sample;

  explicit Cauchy(double location = 0, double scale = 1)
      : location_(location), scale_(scale) {}

  static double draw(
      boost::random::xoshiro256pp& engine,
      double location = 0,
      double scale = 1) {
    thread_local static boost::random::cauchy_distribution<> dist;
    return location + (scale * dist(engine));
  }

  Eigen::VectorXd operator()(
      boost::random::xoshiro256pp& engine, std::size_t n) const {
    return Cauchy::sample(engine, n, location_, scale_);
  }

  Eigen::VectorXd operator()(std::size_t n) const {
    return Cauchy::sample(Xoshiro256pp::getInstance(), n, location_, scale_);
  }

 private:
  double location_;
  double scale_;
};

class Geometric {
 public:
  explicit Geometric(double p = 0.5) : p_(p) {}

  std::size_t sample(boost::random::xoshiro256pp& engine) const {
    return sample(engine, p_);
  }

  static std::size_t sample(
      boost::random::xoshiro256pp& engine, double p = 0.5) {
    thread_local static boost::random::geometric_distribution<> dist;
    thread_local static double dist_p = 0.5;

    if (p != dist_p) {
      dist.param(decltype(dist)::param_type(p));
      dist_p = p;
    }

    return dist(engine);
  }

 private:
  double p_;
};

}  // namespace knoop::random
