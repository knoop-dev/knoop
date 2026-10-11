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

#include <knoop/core/random.h>

#include <Eigen/Dense>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/math/distributions.hpp>
#include <functional>

namespace knoop {

template <typename D>
Eigen::VectorXd generateDist(const D& dist, std::size_t n) {
  Eigen::VectorXd random(n);
  random::Uniform::sample(
      random::Xoshiro256pp::getInstance(), random.data(), n);

  std::ranges::transform(random, random.begin(), [dist](double u) {
    return boost::math::quantile(dist, u);
  });

  return random;
}

class RademacherDistribution {
 public:
  explicit RademacherDistribution() = default;

  static constexpr std::string kName = "rademacher";
  static constexpr std::size_t kNumParams = 0;

  static Eigen::VectorXd generate(std::size_t n) {
    Eigen::VectorXd random(n);
    random::Normal::sample(
        random::Xoshiro256pp::getInstance(), random.data(), n);
    random = random.array().sign();
    return random;
  }
};

class UniformDistribution {
 public:
  explicit UniformDistribution(double lo = 0, double hi = 1)
      : lo_(lo), hi_(hi) {}

  static constexpr std::string kName = "uniform";
  static constexpr std::size_t kNumParams = 2;

  Eigen::VectorXd generate(std::size_t n) const {
    Eigen::VectorXd random(n);
    random::Uniform::sample(
        random::Xoshiro256pp::getInstance(), random.data(), n, lo_, hi_);
    return random;
  }

 private:
  double lo_;
  double hi_;
};

// slightly redundant, but ok
class NormalDistribution {
 public:
  explicit NormalDistribution(double mean = 0, double stddev = 1)
      : mean_(mean), stddev_(stddev) {};

  static constexpr std::string kName = "normal";
  static constexpr std::size_t kNumParams = 2;

  Eigen::VectorXd generate(std::size_t n) const {
    Eigen::VectorXd random(n);
    random::Normal::sample(
        random::Xoshiro256pp::getInstance(), random.data(), n, mean_, stddev_);
    return random;
  }

 private:
  double mean_;
  double stddev_;
};

class BetaDistribution {
 public:
  explicit BetaDistribution(double alpha = 1, double beta = 1)
      : dist_(alpha, beta) {}

  static constexpr std::string kName = "beta";
  static constexpr std::size_t kNumParams = 2;

  Eigen::VectorXd generate(std::size_t n) const {
    return generateDist(dist_, n);
  }

 private:
  boost::math::beta_distribution<double> dist_;
};

class ExponentialDistribution {
 public:
  explicit ExponentialDistribution(double lambda) : dist_(lambda) {}

  static constexpr std::string kName = "exponential";
  static constexpr std::size_t kNumParams = 1;

  Eigen::VectorXd generate(std::size_t n) const {
    return generateDist(dist_, n);
  }

 private:
  boost::math::exponential_distribution<double> dist_;
};

class GammaDistribution {
 public:
  explicit GammaDistribution(double shape, double scale)
      : dist_(shape, scale) {}

  static constexpr std::string kName = "gamma";
  static constexpr std::size_t kNumParams = 2;

  Eigen::VectorXd generate(std::size_t n) const {
    return generateDist(dist_, n);
  }

 private:
  boost::math::gamma_distribution<double> dist_;
};

class LaplaceDistribution {
 public:
  explicit LaplaceDistribution(double location, double scale)
      : dist_(location, scale) {}

  static constexpr std::string kName = "laplace";
  static constexpr std::size_t kNumParams = 2;

  Eigen::VectorXd generate(std::size_t n) const {
    return generateDist(dist_, n);
  }

 private:
  boost::math::laplace_distribution<double> dist_;
};

class StudentsTDistribution {
 public:
  explicit StudentsTDistribution(double df) : dist_(df) {}

  static constexpr std::string kName = "students_t";
  static constexpr std::size_t kNumParams = 1;

  Eigen::VectorXd generate(std::size_t n) const {
    return generateDist(dist_, n);
  }

 private:
  boost::math::students_t_distribution<double> dist_;
};

struct Distribution {
  std::string name;
  std::vector<double> params;
  std::function<Eigen::VectorXd(std::size_t)> fn;

  Eigen::VectorXd operator()(std::size_t n) const { return fn(n); }
};

template <typename Dist, typename... Params>
inline Distribution makeDistribution(Params&&... params) {
  if constexpr (std::is_same_v<Dist, RademacherDistribution>) {
    return Distribution{
        .name = Dist::Name, .params = {}, .fn = [](std::size_t n) {
          return RademacherDistribution::generate(n);
        }};
  } else {
    if (sizeof...(params) != Dist::kNumParams) {
      throw std::runtime_error(
          std::format(
              "Distribution {} requires {} parameters",
              Dist::kName,
              Dist::kNumParams));
    }
    Dist dist = Dist(std::forward<Params>(params)...);
    return Distribution{
        .name = Dist::kName,
        .params = {static_cast<double>(params)...},
        .fn = [dist](std::size_t n) { return dist.generate(n); }};
  }
}

inline Distribution strToDistribution(
    const std::string& name, const std::vector<double>& params) {
  if (name == "uniform")
    return makeDistribution<UniformDistribution>(params[0], params[1]);
  if (name == "beta")
    return makeDistribution<BetaDistribution>(params[0], params[1]);
  if (name == "exponential")
    return makeDistribution<ExponentialDistribution>(params[0]);
  if (name == "gamma")
    return makeDistribution<GammaDistribution>(params[0], params[1]);
  if (name == "normal")
    return makeDistribution<NormalDistribution>(params[0], params[1]);
  if (name == "laplace")
    return makeDistribution<LaplaceDistribution>(params[0], params[1]);
  if (name == "students_t")
    return makeDistribution<StudentsTDistribution>(params[0]);
  throw std::runtime_error(std::format("Unknown distribution {}", name));
}

}  // namespace knoop
