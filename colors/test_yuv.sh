#!/bin/bash

# Parameters
NAME="image_yuv_422"
EXTENSION="yuv"
MODE="--colorspace=yuv --sampling-factor=4:2:2"

# Run test
source `dirname $0`/test_common.sh
