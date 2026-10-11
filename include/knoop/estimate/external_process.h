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

#include <knoop/core/log.h>
#include <knoop/core/utils.h>

#include <boost/asio.hpp>
#include <boost/process.hpp>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace knoop::process {

namespace bp = boost::process;
namespace asio = boost::asio;

struct ProcessResult {
  int exit_code;
  std::string output;
};

// resolve an executable to be run with bp::process
inline bp::filesystem::path resolveExecutable(const std::string& s) {
  bp::filesystem::path exec(s);

  // if it's a path, make sure it exists and is an executable (has +x privilege)
  if (exec.has_parent_path()) {
    bp::filesystem::path exec_abs = bp::filesystem::absolute(exec);
    if (!bp::filesystem::exists(exec_abs) ||
        access(exec_abs.c_str(), X_OK) != 0)
      throw std::runtime_error(
          std::format("{} is not an executable file", exec_abs.string()));
    return exec_abs;
  }

  // otherwise, search PATH for the executable
  auto found = bp::environment::find_executable(exec);
  if (found.empty())
    throw std::runtime_error(
        std::format("Executable {} not found in PATH", exec.string()));
  return found;
}

inline ProcessResult runProcess(
    const bp::filesystem::path& cmd, const std::vector<std::string>& args) {
  asio::io_context ctx;
  asio::readable_pipe pipe{ctx};
  std::string pipe_buf;

  // launch the process
  bp::process proc(ctx, cmd, args, bp::process_stdio{.out = pipe, .err = pipe});

  // read in stdout + stderr
  boost::system::error_code ec;
  asio::read(pipe, asio::dynamic_buffer(pipe_buf), ec);

  if (ec && (ec != asio::error::eof))
    throw std::runtime_error(
        std::format(
            "process {} failed to parse stdout / stderr output", cmd.string()));

  proc.wait();

  return ProcessResult{.exit_code = proc.exit_code(), .output = pipe_buf};
}

}  // namespace knoop::process
