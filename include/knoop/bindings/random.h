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

#define SOL_ALL_SAFETIES_ON 1
#include <knoop/core/random.h>

#include <sol/sol.hpp>

namespace knoop::lua {

/** \brief Register probability distributions to the Lua interface */
inline void defineDistributions(sol::state& lua) {
  lua.set_function(
      "seed", [](std::uint64_t seed) { random::Xoshiro256pp::setSeed(seed); });

  lua.new_usertype<random::Uniform>(
      "uniform",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            return random::Uniform(
                tbl.get_or("lo", 0.0), tbl.get_or("hi", 1.0));
          },
          [](double lo, double hi) { return random::Uniform(lo, hi); }),
      sol::meta_function::call,
      sol::resolve<Eigen::VectorXd(std::size_t) const>(
          &random::Uniform::operator()));

  lua.new_usertype<random::Rademacher>(
      "rademacher",
      sol::call_constructor,
      sol::factories(
          [](const sol::table&) { return random::Rademacher(); },
          []() { return random::Rademacher(); }),
      sol::meta_function::call,
      sol::resolve<Eigen::VectorXd(std::size_t) const>(
          &random::Rademacher::operator()));

  lua.new_usertype<random::Normal>(
      "normal",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            return random::Normal(
                tbl.get_or("location", 0.0), tbl.get_or("scale", 1.0));
          },
          [](double location, double scale) {
            return random::Normal(location, scale);
          }),
      sol::meta_function::call,
      sol::resolve<Eigen::VectorXd(std::size_t) const>(
          &random::Normal::operator()));

  lua.new_usertype<random::Beta>(
      "beta",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            return random::Beta(
                tbl.get_or("shape1", 4.0), tbl.get_or("shape2", 4.0));
          },
          [](double shape1, double shape2) {
            return random::Beta(shape1, shape2);
          }),
      sol::meta_function::call,
      sol::resolve<Eigen::VectorXd(std::size_t) const>(
          &random::Beta::operator()));

  lua.new_usertype<random::Gamma>(
      "gamma",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            return random::Gamma(
                tbl.get_or("shape", 1.0), tbl.get_or("scale", 1.0));
          },
          [](double shape, double scale) { return random::Gamma(shape, scale); }),
      sol::meta_function::call,
      sol::resolve<Eigen::VectorXd(std::size_t) const>(
          &random::Gamma::operator()));

  lua.new_usertype<random::Laplace>(
      "laplace",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            return random::Laplace(
                tbl.get_or("location", 0.0), tbl.get_or("scale", 1.0));
          },
          [](double location, double scale) {
            return random::Laplace(location, scale);
          }),
      sol::meta_function::call,
      sol::resolve<Eigen::VectorXd(std::size_t) const>(
          &random::Laplace::operator()));

  lua.new_usertype<random::Cauchy>(
      "cauchy",
      sol::call_constructor,
      sol::factories(
          [](const sol::table& tbl) {
            return random::Cauchy(
                tbl.get_or("location", 0.0), tbl.get_or("scale", 1.0));
          },
          [](double location, double scale) {
            return random::Cauchy(location, scale);
          }),
      sol::meta_function::call,
      sol::resolve<Eigen::VectorXd(std::size_t) const>(
          &random::Cauchy::operator()));
}

}  // namespace knoop::lua
