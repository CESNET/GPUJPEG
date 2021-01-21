#!/bin/bash

# Parameters
export NAME="image_rgb_444"
export EXTENSION="rgb"
export MODE="--colorspace=rgb  --pixel-format=444-u8-p012"
export FF_FORMAT="rgb24"

# Run test
`dirname $0`/test_common.sh "$@"
