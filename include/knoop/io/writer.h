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

#include <boost/math/distributions/students_t.hpp>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <format>
#include <functional>
#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace knoop {

using Job = std::function<void()>;

class HDF5Writer {
 public:
  HDF5Writer(const HDF5Writer&) = delete;
  HDF5Writer& operator=(const HDF5Writer&) = delete;

  static HDF5Writer& get_instance(
      const std::filesystem::path& results_path,
      std::size_t n_gens,
      std::size_t n_reps) {
    static HDF5Writer instance(results_path, n_gens, n_reps);
    instance_ = &instance;
    return *instance_;
  }

  static HDF5Writer& get_instance() {
    if (instance_ == nullptr)
      throw std::runtime_error("Must initialise Writer before calling");
    return *instance_;
  }

  void write_params(const Params& params);

  template <typename T>
  static void create(
      T&& data,
      const std::string& name,
      const std::vector<std::string>& row_labels,
      const std::vector<std::string>& col_labels);

  template <typename T>
  static void write(
      T&& data, const std::string& name, std::size_t rep, std::size_t gen);

  ~HDF5Writer() {
    {
      std::lock_guard<std::mutex> lg(mutex_);
      done_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();

    summarise();
  }

 private:
  explicit HDF5Writer(
      const std::filesystem::path& results_path,
      std::size_t n_gens,
      std::size_t n_reps)
      : file_(results_path, HighFive::File::Overwrite),
        n_gens_(n_gens),
        n_reps_(n_reps) {
    thread_ = std::thread(&HDF5Writer::threadCallback, this);
    file_.createGroup("params");
    file_.createGroup("raw");
    file_.createGroup("summary");
  }

  template <typename T>
  void createEstimator(
      T&& data,
      const std::string& name,
      const std::vector<std::string>& row_labels,
      const std::vector<std::string>& col_labels);

  template <typename T>
  void writeEstimator(
      T&& data, const std::string& name, std::size_t rep, std::size_t gen);

  void summarise();

  inline static HDF5Writer* instance_ = nullptr;

  HighFive::File file_;
  std::size_t n_gens_;
  std::size_t n_reps_;

  std::unordered_set<std::string> datasets_;
  std::deque<Job> jobs_;
  std::condition_variable cv_;
  std::thread thread_;
  std::mutex mutex_;
  bool done_ = false;

  void threadCallback();
};

inline void HDF5Writer::threadCallback() {
  std::unique_lock<std::mutex> lk(mutex_);
  while (!done_ || !jobs_.empty()) {
    cv_.wait(lk, [this] { return done_ || !jobs_.empty(); });

    while (!jobs_.empty()) {
      Job job = std::move(jobs_.front());
      jobs_.pop_front();
      lk.unlock();
      job();
      lk.lock();
    }
  }
}

template <typename T>
inline void HDF5Writer::createEstimator(
    T&& /*data*/,
    const std::string& name,
    const std::vector<std::string>& row_labels,
    const std::vector<std::string>& col_labels) {
  {
    std::lock_guard<std::mutex> lg(mutex_);

    jobs_.push_back([this,
                     name_raw = std::format("raw/{}", name),
                     row_copy = row_labels,
                     col_copy = col_labels]() {
      if (!file_.exist(name_raw)) file_.createGroup(name_raw);

      HighFive::Group estimator = file_.getGroup(name_raw);

      if (!estimator.hasAttribute("row_labels"))
        estimator.createAttribute("row_labels", row_copy);

      if (!estimator.hasAttribute("col_labels"))
        estimator.createAttribute("col_labels", col_copy);
    });
  }
  cv_.notify_one();
}

template <typename T>
inline void HDF5Writer::writeEstimator(
    T&& data, const std::string& name, std::size_t rep, std::size_t gen) {
  {
    std::lock_guard<std::mutex> lg(mutex_);

    jobs_.push_back(
        [this,
         copy = data,
         dir = std::format("raw/{}/rep{:03d}", name, rep + 1),
         path = std::format("raw/{}/rep{:03d}/gen{:03d}", name, rep + 1, gen)] {
          if (!file_.exist(dir)) file_.createGroup(dir);
          file_.createDataSet(path, copy);
        });

    if (!datasets_.contains(name)) datasets_.insert(name);
  }
  cv_.notify_one();
}

inline void HDF5Writer::write_params(const Params& params) {
  auto params_group = file_.getGroup("params");

  // assemble effect matrix
  Eigen::MatrixXd effect_matrix(params.geno.n_loc, params.pheno.n_pheno);
  effect_matrix.setZero();

  for (std::size_t pheno = 0; pheno < params.pheno.n_pheno; ++pheno) {
    auto effects = params.pheno.pheno_effects[pheno];
    auto locs = params.pheno.pheno_loc[pheno];
    for (std::size_t el = 0; el < locs.size(); ++el)
      effect_matrix(locs[el], pheno) = effects(el);
  }

  file_.createGroup("params/genome");
  file_.createDataSet("params/genome/n_loc", params.geno.n_loc);
  file_.createDataSet("params/genome/v_maf", params.geno.v_maf);
  file_.createDataSet("params/genome/v_rec", params.geno.v_rec);
  file_.createDataSet("params/genome/v_mut", params.geno.v_mut);

  file_.createGroup("params/phenome");
  file_.createDataSet("params/phenome/n_pheno", params.pheno.n_pheno);
  file_.createDataSet("params/phenome/names", params.pheno.names);
  file_.createDataSet("params/phenome/n_locs", params.pheno.n_locs);
  file_.createDataSet("params/phenome/effects", effect_matrix);
  file_.createDataSet("params/phenome/gen_cor", params.pheno.gen_cor);
  file_.createDataSet("params/phenome/env_cor", params.pheno.env_cor);
  file_.createDataSet("params/phenome/h2_gen", params.pheno.var_gen);
  file_.createDataSet("params/phenome/h2_env", params.pheno.var_env);
  file_.createDataSet("params/phenome/h2_vert", params.pheno.var_vert);

  file_.createGroup("params/mating");
  file_.createDataSet("params/mating/mate_cor", params.mate.mate_cor);
  file_.createDataSet("params/mating/tol_inf", params.mate.tol_inf);
  file_.createDataSet("params/mating/max_itr", params.mate.max_itr);
  file_.createDataSet("params/mating/temp_init", params.mate.temp_init);
  file_.createDataSet("params/mating/temp_decay", params.mate.temp_decay);

  file_.createGroup("params/simulation");
  file_.createDataSet("params/simulation/n_ind", params.global.n_ind);
  file_.createDataSet("params/simulation/n_gens", params.global.n_gens);
  file_.createDataSet("params/simulation/rng_seed", params.global.rng_seed);
}

template <typename T>
inline void HDF5Writer::create(
    T&& data,
    const std::string& name,
    const std::vector<std::string>& row_labels,
    const std::vector<std::string>& col_labels) {
  HDF5Writer::get_instance().createEstimator(data, name, row_labels, col_labels);
}

template <typename T>
inline void HDF5Writer::write(
    T&& data, const std::string& name, std::size_t rep, std::size_t gen) {
  HDF5Writer::get_instance().writeEstimator(data, name, rep, gen);
}

inline void HDF5Writer::summarise() {
  for (const std::string& name : datasets_) {
    std::string path_summary = std::format("summary/{}", name);

    // the group we want to write to
    file_.createGroup(path_summary);

    // get the labels
    std::string path_raw = std::format("raw/{}", name);
    auto group = file_.getGroup(path_raw);
    auto row_labels =
        group.getAttribute("row_labels").read<std::vector<std::string>>();
    auto col_labels =
        group.getAttribute("col_labels").read<std::vector<std::string>>();

    // infer the dimension
    Eigen::MatrixXd m;
    file_.getDataSet(std::format("{}/rep{:03d}/gen{:03d}", path_raw, 1, 1))
        .read(m);

    std::size_t n_rows = m.rows();
    std::size_t n_cols = m.cols();

    Eigen::MatrixXd mean = Eigen::MatrixXd::Zero(n_rows, n_cols);
    Eigen::MatrixXd stderr = Eigen::MatrixXd(n_rows, n_cols);

    auto ds_mean = file_.createDataSet<double>(
        std::format("{}/mean", path_summary),
        HighFive::DataSpace({n_gens_, n_rows, n_cols}));

    auto ds_stderr = file_.createDataSet<double>(
        std::format("{}/stderr", path_summary),
        HighFive::DataSpace({n_gens_, n_rows, n_cols}));

    auto ds_confint_lo = file_.createDataSet<double>(
        std::format("{}/confint_lo", path_summary),
        HighFive::DataSpace({n_gens_, n_rows, n_cols}));

    auto ds_confint_hi = file_.createDataSet<double>(
        std::format("{}/confint_hi", path_summary),
        HighFive::DataSpace({n_gens_, n_rows, n_cols}));

    for (auto* ds : {&ds_mean, &ds_stderr, &ds_confint_lo, &ds_confint_hi}) {
      ds->createAttribute("row_labels", row_labels);
      ds->createAttribute("col_labels", col_labels);
    }

    double t = 1;

    if (n_reps_ > 1) {
      boost::math::students_t_distribution dist(n_reps_ - 1);
      t = boost::math::quantile(dist, 0.975);
    }

    for (std::size_t gen = 0; gen < n_gens_; ++gen) {
      mean.setZero();
      stderr.setZero();

      for (std::size_t rep = 0; rep < n_reps_; ++rep) {
        file_
            .getDataSet(
                std::format(
                    "{}/rep{:03d}/gen{:03d}", path_raw, rep + 1, gen + 1))
            .read(m);
        mean += m;
      }
      mean /= static_cast<double>(n_reps_);

      for (std::size_t rep = 0; rep < n_reps_; ++rep) {
        file_
            .getDataSet(
                std::format(
                    "{}/rep{:03d}/gen{:03d}", path_raw, rep + 1, gen + 1))
            .read(m);
        stderr += (m - mean).array().square().matrix();
      }
      stderr /= static_cast<double>(n_reps_ - 1);
      stderr = (stderr / n_reps_).array().sqrt();

      Eigen::MatrixXd confint_lo = mean - t * stderr;
      Eigen::MatrixXd confint_hi = mean + t * stderr;

      // write the mean and variance matrices
      ds_mean.select({gen, 0, 0}, {1, n_rows, n_cols})
          .reshapeMemSpace({n_rows, n_cols})
          .write(mean);

      ds_stderr.select({gen, 0, 0}, {1, n_rows, n_cols})
          .reshapeMemSpace({n_rows, n_cols})
          .write(stderr);

      ds_confint_lo.select({gen, 0, 0}, {1, n_rows, n_cols})
          .reshapeMemSpace({n_rows, n_cols})
          .write(confint_lo);

      ds_confint_hi.select({gen, 0, 0}, {1, n_rows, n_cols})
          .reshapeMemSpace({n_rows, n_cols})
          .write(confint_hi);
    }
  }
}

}  // namespace knoop
