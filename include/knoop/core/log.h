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

#include <boost/algorithm/string.hpp>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

namespace knoop {

enum LogLevel { Debug, Info, Warning, Error, None };

inline std::string logLevelToString(LogLevel level) {
  switch (level) {
    case LogLevel::Debug:
      return "DEBUG";
    case LogLevel::Info:
      return "INFO";
    case LogLevel::Warning:
      return "WARNING";
    case LogLevel::Error:
      return "ERROR";
    case LogLevel::None:
      return "NONE";
  }
  __builtin_unreachable();
}

inline LogLevel logLevelFromString(const std::string& level) {
  std::string upper = boost::to_upper_copy(level);
  if (upper == "DEBUG") return LogLevel::Debug;
  if (upper == "INFO") return LogLevel::Info;
  if (upper == "WARNING") return LogLevel::Warning;
  if (upper == "ERROR") return LogLevel::Error;
  if (upper == "NONE") return LogLevel::None;
  throw std::runtime_error("Unknown LogLevel " + upper);
}

inline std::ostream& operator<<(std::ostream& os, LogLevel level) {
  return os << logLevelToString(level);
}

class Log {
 public:
  Log(const Log&) = delete;
  Log& operator=(const Log&) = delete;

  static Log& getInstance(std::ostream& out, const LogLevel level) {
    static Log instance(out, level);
    return instance;
  }

  static Log& getInstance() { return getInstance(std::cout, LogLevel::Info); }

  ~Log() {
    {
      std::lock_guard<std::mutex> lg(mutex_);
      done_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  static void file(const std::filesystem::path& path, LogLevel level);
  static void stream(std::ostream& stream, LogLevel level);

  template <typename... Args>
  void log(LogLevel level, std::format_string<Args...>, Args&&... args);

  template <typename... Args>
  static void debug(std::format_string<Args...>, Args&&... args);

  template <typename... Args>
  static void info(std::format_string<Args...>, Args&&... args);

  template <typename... Args>
  static void warning(std::format_string<Args...>, Args&&... args);

  template <typename... Args>
  static void error(std::format_string<Args...>, Args&&... args);

 private:
  explicit Log(
      std::ostream& out = std::cout, const LogLevel level = LogLevel::Info)
      : stream_(out), level_(level) {
    thread_ = std::thread(&Log::threadCallback, this);
  }

  std::ostream& stream_;
  std::deque<std::string> messages_;
  std::condition_variable cv_;
  std::thread thread_;
  std::mutex mutex_;
  LogLevel level_;
  bool done_ = false;

  void threadCallback();
  static std::string getTimeStr();
  static std::string formatMsg(const std::string& msg, LogLevel level);
};

inline void Log::threadCallback() {
  std::unique_lock<std::mutex> lk(mutex_);
  while (!done_ || !messages_.empty()) {
    cv_.wait(lk, [this] { return done_ || !messages_.empty(); });

    while (!messages_.empty()) {
      std::string msg = std::move(messages_.front());
      messages_.pop_front();

      lk.unlock();
      stream_ << msg << "\n";
      lk.lock();
    }
  }
}

inline std::string Log::getTimeStr() {
  const auto now = std::chrono::system_clock::now();
  const auto tse = now.time_since_epoch();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(tse) % 1000;
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_now;

  localtime_r(&t, &tm_now);

  std::ostringstream oss;

  oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
      << std::setw(3) << ms.count();

  return oss.str();
}

inline std::string Log::formatMsg(
    const std::string& msg, const LogLevel level) {
  std::string time_str = getTimeStr();
  std::ostringstream prefix_stream;

  prefix_stream << "[" << logLevelToString(level) << "] "
                << "[" << time_str << "] " << "[thread "
                << std::this_thread::get_id() << "] ";
  std::string prefix = prefix_stream.str();

  std::string body = msg;
  boost::replace_all(body, "\n", "\n" + prefix);

  return prefix + body;
}

template <typename... Args>
inline void Log::log(
    LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
  if (level < level_) return;
  {
    std::string msg = std::format(fmt, std::forward<Args>(args)...);
    std::lock_guard<std::mutex> lg(mutex_);
    std::string msg_format = formatMsg(msg, level);
    messages_.push_back(msg_format);
  }
  cv_.notify_one();
}

inline void Log::file(const std::filesystem::path& path, LogLevel log_level) {
  static std::fstream log_file(path, std::ios::out);
  Log::getInstance(log_file, log_level);
}

inline void Log::stream(std::ostream& stream, LogLevel log_level) {
  Log::getInstance(stream, log_level);
}

template <typename... Args>
inline void Log::debug(std::format_string<Args...> fmt, Args&&... args) {
  Log::getInstance().log(LogLevel::Debug, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void Log::info(std::format_string<Args...> fmt, Args&&... args) {
  Log::getInstance().log(LogLevel::Info, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void Log::warning(std::format_string<Args...> fmt, Args&&... args) {
  Log::getInstance().log(LogLevel::Warning, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void Log::error(std::format_string<Args...> fmt, Args&&... args) {
  Log::getInstance().log(LogLevel::Error, fmt, std::forward<Args>(args)...);
}

}  // namespace knoop
