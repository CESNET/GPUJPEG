#!/bin/bash

declare -A TESTS
i=1
RET=0

TESTS[$i,NAME]="image_yuv_444p"
TESTS[$i,EXTENSION]="yuv"
TESTS[$i,MODE]="--colorspace=ycbcr-bt601 --pixel-format=444-u8-p0p1p2"
TESTS[$i,FF_FORMAT]="yuv444p"
i=$(($i+1))

# Parameters
TESTS[$i,NAME]="image_rgb_444"
TESTS[$i,EXTENSION]="rgb"
TESTS[$i,MODE]="--colorspace=rgb --pixel-format=444-u8-p012"
TESTS[$i,FF_FORMAT]="rgb24"
i=$(($i+1))

for n in `seq $(($i-1))`; do
        # Parameters
        export NAME=${TESTS[$n,NAME]}
        export EXTENSION=${TESTS[$n,EXTENSION]}
        export MODE=${TESTS[$n,MODE]}
        export FF_FORMAT=${TESTS[$n,FF_FORMAT]}

        # Run test
        echo "Running test $NAME"
        `dirname $0`/test_common.sh "$@"
        RC=$?
        if [ $RC -ne 0 ]; then
                RET=$RC
        fi
done

exit $RET

