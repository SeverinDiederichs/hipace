#! /usr/bin/env bash

# This file is part of the Hipace++ test suite.
# It runs a Hipace simulation in the blowout regime and compares the result
# with SI units.

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/beam_in_vacuum
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/inputs_normalized \
        amr.n_cell = 32 32 10 \
        max_step = 20 \
        geometry.prob_lo = -2. -2. -2. \
        geometry.prob_hi =  2.  2.  2. \
        hipace.dt = 3. \
        hipace.output_period = 20 \
        beam.density = 1.e-8 \
        beam.radius = 1. \
        beam.ppc = 4 4 1 \
        hipace.external_ExmBy_slope = .5 \
        hipace.file_prefix = $TEST_NAME

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_beam_push.py --output-dir=$TEST_NAME

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME
