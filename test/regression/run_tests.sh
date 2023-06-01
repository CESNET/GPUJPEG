#!/bin/sh -eu

DIR=$(dirname $0)
GPUJPEG=${1:-$DIR/../../gpujpegtool}
if [ ! -x "$GPUJPEG" ] && [ -x "$DIR/../../build/gpujpegtool" ]; then
        GPUJPEG=$DIR/../../build/gpujpegtool
fi
REQUESTED_PSNR=50

test_commit_b620be2() {
        $GPUJPEG -e -s 1920x1080 -r 1 -f 444-u8-p0p1p2 /dev/zero out.jpg
        $GPUJPEG -d out.jpg out.rgb
        SIZE=$(stat -c %s out.rgb)
        dd if=/dev/zero bs=$SIZE count=1 of=in.rgb
        PSNR=`compare -metric PSNR -depth 8 -size 1920x1080 out.rgb in.rgb null: 2>&1 || true`
        echo PSNR: $PSNR
        if [ $PSNR != inf ] && expr $PSNR != 0 && expr $PSNR \< $REQUESTED_PSNR; then
                echo " black pattern doesn't match!\n" >&2
                exit 1
        fi

        $GPUJPEG -e -s 16x16 -r 1 -f u8 /dev/zero out.jpg
        $GPUJPEG -d out.jpg out.r
        SIZE=$(stat -c %s out.r)
        dd if=/dev/zero bs=$SIZE count=1 of=in.r
        PSNR=`compare -metric PSNR -depth 8 -size 16x16 gray:out.r gray:in.r null: 2>&1 || true`
        echo PSNR: $PSNR
        if [ $PSNR != inf ] && expr $PSNR != 0 && expr $PSNR \< $REQUESTED_PSNR; then
                echo "grayscale pattern doesn't match!\n" >&2
                exit 1
        fi

        rm in.rgb in.r out.rgb out.r out.jpg
}

# currently just a simple read/write tests without validating file contents
test_pam_pnm_y4m() {
        readonly w=256
        readonly h=256
        printf "YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C444 XCOLORRANGE=FULL\nFRAME\n" $w $h > in.y4m
        dd if=/dev/zero of=data.raw bs=$((w*h*3)) count=1
        cat data.raw >> in.y4m
        $GPUJPEG -e in.y4m out.jpg
        $GPUJPEG -d out.jpg out.y4m
        $GPUJPEG -d out.jpg out.pam
        $GPUJPEG -d out.jpg out.pnm
        $GPUJPEG -e out.pam out.jpg
        $GPUJPEG -e out.pnm out.jpg
        rm data.raw in.y4m out.jpg out.pam out.pnm out.y4m
}

test_commit_b620be2
test_pam_pnm_y4m

