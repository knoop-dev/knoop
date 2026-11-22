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

#include <getopt.h>
#include <knoop/bindings.h>
#include <knoop/core.h>
#include <knoop/estimate.h>
#include <knoop/init.h>
#include <knoop/io.h>
#include <knoop/simulation.h>
#include <wordexp.h>

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

constexpr std::string getOSName() {
  std::string os = knoop_OS;
  if (os == "Darwin") return "macOS";
  return os;
}

void displayLine(std::size_t n_chars, bool newline = true) {
  std::string line;
  for (std::size_t el = 0; el < n_chars; ++el) line += "\u2500";
  std::cout << line << (newline ? "\n" : "");
}

void displayVersion() {
  std::string version_str = std::format(
      "knoop v{} {} {} ({} build, {})",
      knoop_VERSION,
      getOSName(),
      knoop_ARCH,
      knoop_BUILD_TYPE,
      __DATE__);

  std::cout << version_str << std::endl;
}

void displayHeader() {
  std::string header_str = std::format(
      "{:<40}{:>40}\n{:<40}{:>40}",
      std::format(
          "\u25c7 knoop v{} {} {} ({} build)",
          knoop_VERSION,
          getOSName(),
          knoop_ARCH,
          knoop_BUILD_TYPE),
      "https://github.com/lvthnn/knoop",
      "\u00a9 2025-2026 Kári Hlynsson",
      "GNU General Public License v3");

  displayLine(82);
  std::cout << header_str << std::endl;
  displayLine(82);
}

void displayHelp() {
  std::string help_str = R"(
Usage: knoop <COMMAND> [OPTIONS]

Commands:
  -h, --help      Display this message and exit
  -v, --version   Display version information and exit

For more information on subcommands, run knoop help <COMMAND>.
  )";

  std::cout << help_str << std::endl;
}

int main(int argc, char* argv[]) {
  sol::state sol = knoop::lua::getLuaState();

  if (argc == 1) {
    displayHeader();
    displayHelp();
  }
  if (argc > 1) {
    std::string lua_file = std::string(argv[1]);
    knoop::lua::runKnoopScript(sol, lua_file);
  }
}
