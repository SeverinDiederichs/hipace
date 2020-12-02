#! /usr/bin/env bash

# This file is part of the Hipace test suite.
# It runs a Hipace simulation in the blowout regime and compares the result
# with SI units.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

rm -rf negative_gradient_data
rm -rf positive_gradient_data

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        amr.n_cell = 32 32 32 \
        max_step = 20 \
        geometry.prob_lo = -2. -2. -2. \
        geometry.prob_hi =  2.  2.  2. \
        hipace.slice_beam = 1 \
        hipace.3d_on_host = 1 \
        hipace.dt = 0. \
        hipace.output_period = 20 \
        beam.density = 1.e-8 \
        beam.radius = 1. \
        beam.ppc = 4 4 1 \
        hipace.external_accel_field_strength = -.5 \
        hipace.verbose=1\
        hipace.do_adaptive_time_step=1\
        plasma.density=1 \
        > negative_gradient.txt

# mv plt00001 negative_gradient_data

mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        amr.n_cell = 32 32 32 \
        max_step = 20 \
        geometry.prob_lo = -2. -2. -2. \
        geometry.prob_hi =  2.  2.  2. \
        hipace.slice_beam = 1 \
        hipace.3d_on_host = 1 \
        hipace.dt = 0. \
        hipace.output_period = 20 \
        beam.density = 1.e-8 \
        beam.radius = 1. \
        beam.ppc = 4 4 1 \
        hipace.external_accel_field_strength = .5 \
        hipace.verbose=1\
        hipace.do_adaptive_time_step=1\
        plasma.density=1 \
        > positive_gradient.txt

# mv plt00001 positive_gradient_data

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_adaptive_ts.py

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --plotfile plt00020 \
    --test-name adaptive_time_step.1Rank