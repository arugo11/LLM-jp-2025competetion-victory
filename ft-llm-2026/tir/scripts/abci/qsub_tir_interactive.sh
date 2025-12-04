#!/bin/bash
set -eux

qsub "$(dirname "$0")/run_tir_interactive.sh"
