#!/bin/bash

declare -A TESTS
i=1
RET=0

TESTS[$i,NAME]="image_yuv_444p_subsampled"
TESTS[$i,EXTENSION]="yuv"
TESTS[$i,MODE]="--colorspace=ycbcr-bt601 --subsampled --pixel-format=444-u8-p0p1p2"
TESTS[$i,FF_FORMAT]="yuv444p"
i=$(($i+1))

TESTS[$i,NAME]="image_yuv_422_interleaved"
TESTS[$i,EXTENSION]="yuv"
TESTS[$i,MODE]="--pixel-format=422-u8-p1020 -i"
TESTS[$i,FF_FORMAT]="uyvy422"
i=$(($i+1))

TESTS[$i,NAME]="image_yuv_420p_native_709"
TESTS[$i,EXTENSION]="yuv"
TESTS[$i,MODE]="-N --colorspace ycbcr-bt709 --pixel-format=420-u8-p0p1p2"
TESTS[$i,FF_FORMAT]="yuv420p"
i=$(($i+1))

TESTS[$i,NAME]="image_rgb_444"
TESTS[$i,EXTENSION]="rgb"
TESTS[$i,MODE]="--colorspace=rgb --pixel-format=444-u8-p012"
TESTS[$i,FF_FORMAT]="rgb24"
i=$(($i+1))

TESTS[$i,NAME]="image_rgb_444_native"
TESTS[$i,EXTENSION]="rgb"
TESTS[$i,MODE]="-N --pixel-format=444-u8-p012"
TESTS[$i,FF_FORMAT]="rgb24"
i=$(($i+1))

TESTS[$i,NAME]="image_rgb0_444_interleaved_subsampled"
TESTS[$i,EXTENSION]="rgba"
TESTS[$i,MODE]="-i -S -f 4444-u8-p0123"
TESTS[$i,FF_FORMAT]="rgba"
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
                tput setaf 1 2>/dev/null || true # red
                echo "Test $NAME failed!!!"
                tput sgr0 # reset
                RET=$RC
        fi
done

exit $RET

