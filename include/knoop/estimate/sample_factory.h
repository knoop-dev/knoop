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

#include <knoop/estimate/external.h>
#include <knoop/estimate/greml.h>
#include <knoop/estimate/gwas.h>
#include <knoop/estimate/haseman_elston.h>
#include <knoop/estimate/sample.h>
#include <knoop/sample/proband.h>

namespace knoop {

template <ProbandType P>
inline SampleEstimator<P> buildSampleEstimator(
    const SampleEstimatorSpec& spec) {
  if (spec.type == "external") {
    return sampleExternal<P>(
        spec.name,
        spec.exec.value(),
        spec.exec_args,
        spec.n_rows.value(),
        spec.n_cols.value(),
        spec.row_names,
        spec.col_names);
  }
  if (spec.type == "sample-mean") return sampleMean<P>();
  if (spec.type == "sample-var") return sampleVar<P>();
  if (spec.type == "sample-cov") return sampleCov<P>();
  if (spec.type == "sample-mate-cor") return sampleMateCor<P>();
  if (spec.type == "haseman-elston") return sampleHasemanElston<P>(spec.name);
  if (spec.type == "greml") return sampleGREML<P>(spec.name);
  if (spec.type == "gwas") {
    std::size_t n_pcs = 0;
    double pval_threshold = 5e-8;
    if (spec.params.size() == 1) n_pcs = std::stoull(spec.params[0]);
    if (spec.params.size() == 2) {
      n_pcs = parse<std::size_t>(spec.params[0]);
      pval_threshold = parse<double>(spec.params[1]);
    }
    return sampleGWAS<P>(spec.name, n_pcs, pval_threshold);
  }
  throw std::runtime_error("Unknown sample estimator type: " + spec.type);
}

}  // namespace knoop
