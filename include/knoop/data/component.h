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

#include <boost/algorithm/string/case_conv.hpp>
#include <format>
#include <ostream>
#include <stdexcept>
#include <string>

namespace knoop {

enum Component { Genetic = 0, Environmental = 1, Vertical = 2, Total = 3 };

inline Component operator++(Component& type, int) {
  Component old = type;
  type = (type == Component::Total) ? Component::Genetic
                                    : Component(static_cast<int>(type) + 1);
  return old;
}

inline Component& operator++(Component& type) {
  type = (type == Component::Total) ? Component::Genetic
                                    : Component(static_cast<int>(type) + 1);
  return type;
}

inline Component componentFromString(const std::string& s) {
  std::string l = boost::to_lower_copy(s);
  if (l == "genetic") return Component::Genetic;
  if (l == "environmental") return Component::Environmental;
  if (l == "vertical") return Component::Vertical;
  if (l == "total") return Component::Total;
  throw std::runtime_error(std::format("Unknown component type {}", l));
}

inline std::string componentToString(Component type) {
  switch (type) {
    case Component::Genetic:
      return "genetic";
    case Component::Environmental:
      return "environ";
    case Component::Vertical:
      return "vertical";
    case Component::Total:
      return "total";
  }
  __builtin_unreachable();
}

inline std::ostream& operator<<(std::ostream& os, Component type) {
  return os << componentToString(type);
}

}  // namespace knoop
