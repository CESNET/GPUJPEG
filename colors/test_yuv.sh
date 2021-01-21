#!/bin/bash

# Parameters
export NAME="image_yuv_444"
export EXTENSION="yuv"
export MODE="--colorspace=ycbcr-bt601 --pixel-format=444-u8-p0p1p2"
export FF_FORMAT="yuv444p"

# Run test
`dirname $0`/test_common.sh "$@"
