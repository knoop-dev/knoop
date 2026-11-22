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
 private:
  explicit HDF5Writer(
      const std::filesystem::path& results_path,
      std::size_t n_gens,
      std::size_t n_reps)
      : path_(results_path),
        file_(results_path, HighFive::File::Overwrite),
        n_gens_(n_gens),
        n_reps_(n_reps) {
    thread_ = std::thread(&HDF5Writer::threadCallback, this);
    file_.createGroup("params");
    file_.createGroup("raw");
    file_.createGroup("summary");
  }

  void createEstimator(
      const std::string& name,
      const std::vector<std::string>& row_labels,
      const std::vector<std::string>& col_labels,
      std::size_t n_rows,
      std::size_t n_cols);

  void writeEstimator(
      const Eigen::MatrixXd& data,
      const std::string& name,
      std::size_t rep,
      std::size_t gen);

  void summarise();

  static std::unique_ptr<HDF5Writer> instance_;

  std::filesystem::path path_;
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

 public:
  HDF5Writer(const HDF5Writer&) = delete;
  HDF5Writer& operator=(const HDF5Writer&) = delete;

  static HDF5Writer& getInstance(
      const std::filesystem::path& results_path,
      std::size_t n_gens,
      std::size_t n_reps) {
    /**
     * If the writer has already been instantiated AND it points to the file
     * path that the call asks for, return it
     */
    if (instance_ && instance_->path_ == results_path) return *instance_;

    /**
     * If no writer has been set up, create a new one
     */
    instance_ = std::unique_ptr<HDF5Writer>(
        new HDF5Writer(results_path, n_gens, n_reps));

    return *instance_;
  }

  static HDF5Writer& getInstance() {
    if (!instance_)
      throw KnoopGenericError("Must initialise Writer before calling");
    return *instance_;
  }

  static void shutdown() { instance_.reset(); }

  void writeParams(const Params& params);

  static void create(
      const std::string& name,
      const std::vector<std::string>& row_labels,
      const std::vector<std::string>& col_labels,
      std::size_t n_rows,
      std::size_t n_cols = 1);

  static void write(
      const Eigen::MatrixXd& data,
      const std::string& name,
      std::size_t rep,
      std::size_t gen);

  ~HDF5Writer() {
    {
      std::lock_guard<std::mutex> lg(mutex_);
      done_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();

    summarise();
  }
};

inline std::unique_ptr<HDF5Writer> HDF5Writer::instance_;

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

inline void HDF5Writer::createEstimator(
    const std::string& name,
    const std::vector<std::string>& row_labels,
    const std::vector<std::string>& col_labels,
    std::size_t n_rows,
    std::size_t n_cols) {
  {
    std::lock_guard<std::mutex> lg(mutex_);

    Log::debug(
        "Creating estimator named: {}, n_rows: {}, n_cols: {}",
        name,
        n_rows,
        n_cols);

    jobs_.emplace_back([this,
                     n_rows = n_rows,
                     n_cols = n_cols,
                     name_raw = std::format("raw/{}", name),
                     row_labels = row_labels,
                     col_labels = col_labels] {
      if (!file_.exist(name_raw)) {
        auto ds = file_.createDataSet<double>(
            name_raw, HighFive::DataSpace({n_reps_, n_gens_, n_rows, n_cols}));

        ds.createAttribute("row_labels", row_labels);
        ds.createAttribute("col_labels", col_labels);
      }
    });
  }
  cv_.notify_one();
}

inline void HDF5Writer::writeEstimator(
    const Eigen::MatrixXd& data,
    const std::string& name,
    std::size_t rep,
    std::size_t gen) {
  Log::debug(
      "Writing into estimator: {}, rep: {}, gen: {}", name, rep + 1, gen + 1);

  {
    std::lock_guard<std::mutex> lg(mutex_);

    jobs_.emplace_back([this, rep, gen, name, data] {
      auto ds = file_.getDataSet(std::format("raw/{}", name));
      std::size_t n_rows = data.rows();
      std::size_t n_cols = data.cols();

      ds.select({rep, gen, 0, 0}, {1, 1, n_rows, n_cols})
          .reshapeMemSpace({n_rows, n_cols})
          .write(data);
    });

    if (!datasets_.contains(name)) datasets_.insert(name);
  }
  cv_.notify_one();
}

inline void HDF5Writer::writeParams(const Params& params) {
  auto params_group = file_.getGroup("params");

  // assemble effect matrix
  Eigen::MatrixXd effect_matrix(params.geno.numLoci(), params.pheno.size());
  effect_matrix.setZero();

  for (std::size_t pheno = 0; pheno < params.pheno.size(); ++pheno) {
    auto effects = params.pheno.effects(pheno);
    auto locs = params.pheno.causalLoci(pheno);
    for (std::size_t el = 0; el < locs.size(); ++el)
      effect_matrix(locs[el], pheno) = effects(el);
  }

  file_.createGroup("params/genome");
  file_.createDataSet("params/genome/n_loc", params.geno.numLoci());
  file_.createDataSet(
      "params/genome/locus_freq", params.geno.locusFrequencies());
  file_.createDataSet(
      "params/genome/locus_rec", params.geno.locusRecombination());
  file_.createDataSet("params/genome/locus_mut", params.geno.locusMutation());

  file_.createGroup("params/phenome");
  file_.createDataSet("params/phenome/n_pheno", params.pheno.size());
  file_.createDataSet("params/phenome/names", params.pheno.names());
  file_.createDataSet("params/phenome/n_locs", params.pheno.numCausalLoci());
  file_.createDataSet("params/phenome/effects", effect_matrix);
  file_.createDataSet(
      "params/phenome/gen_cor", params.pheno.geneticComponentCorrelation());
  file_.createDataSet(
      "params/phenome/env_cor", params.pheno.environComponentCorrelation());
  file_.createDataSet("params/phenome/var_gen", params.pheno.varGenetic());
  file_.createDataSet(
      "params/phenome/var_env", params.pheno.varEnvironmental());
  file_.createDataSet("params/phenome/var_vert", params.pheno.varIndirect());

  file_.createGroup("params/mating");
  file_.createDataSet(
      "params/mating/mate_cor", params.mate.spouseCorrelation());
  file_.createDataSet("params/mating/tol_inf", params.mate.tolerance());
  file_.createDataSet("params/mating/max_itr", params.mate.maxIterations());
  file_.createDataSet("params/mating/temp_init", params.mate.temperatureInit());
  file_.createDataSet(
      "params/mating/temp_decay", params.mate.temperatureDecay());

  file_.createGroup("params/simulation");
  file_.createDataSet("params/simulation/n_ind", params.numIndividuals());
  file_.createDataSet("params/simulation/n_gens", params.numGenerations());
  file_.createDataSet("params/simulation/rng_seed", params.randomSeed());
}

inline void HDF5Writer::create(
    const std::string& name,
    const std::vector<std::string>& row_labels,
    const std::vector<std::string>& col_labels,
    std::size_t n_rows,
    std::size_t n_cols) {
  HDF5Writer::getInstance().createEstimator(
      name, row_labels, col_labels, n_rows, n_cols);
}

inline void HDF5Writer::write(
    const Eigen::MatrixXd& data,
    const std::string& name,
    std::size_t rep,
    std::size_t gen) {
  HDF5Writer::getInstance().writeEstimator(data, name, rep, gen);
}

inline void HDF5Writer::summarise() {
  for (const std::string& name : datasets_) {
    std::string path_summary = std::format("summary/{}", name);

    // the group we want to write to
    file_.createGroup(path_summary);

    // get the data set
    auto ds = file_.getDataSet(std::format("raw/{}", name));
    auto dims = ds.getDimensions();
    std::size_t n_rows = dims[2];
    std::size_t n_cols = dims[3];

    auto row_labels =
        ds.getAttribute("row_labels").read<std::vector<std::string>>();
    auto col_labels =
        ds.getAttribute("col_labels").read<std::vector<std::string>>();

    Eigen::MatrixXd m(n_rows, n_cols);
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

    for (auto* ds_stat :
         {&ds_mean, &ds_stderr, &ds_confint_lo, &ds_confint_hi}) {
      ds_stat->createAttribute("row_labels", row_labels);
      ds_stat->createAttribute("col_labels", col_labels);
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
        ds.select({rep, gen, 0, 0}, {1, 1, n_rows, n_cols})
            .reshapeMemSpace({n_rows, n_cols})
            .read(m);
        mean += m;
      }
      mean /= static_cast<double>(n_reps_);

      for (std::size_t rep = 0; rep < n_reps_; ++rep) {
        ds.select({rep, gen, 0, 0}, {1, 1, n_rows, n_cols})
            .reshapeMemSpace({n_rows, n_cols})
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
