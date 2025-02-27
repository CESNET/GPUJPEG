#!/bin/sh -eu

DIR=$(dirname $0)
GPUJPEG=${1:-$DIR/../../gpujpegtool}
if [ ! -x "$GPUJPEG" ] && [ -x "$DIR/../../build/gpujpegtool" ]; then
        GPUJPEG=$DIR/../../build/gpujpegtool
fi

readonly REQUESTED_PSNR=50
readonly BROKEN_IM_THR=110

numeric_compare() {
        rc=$(awk "BEGIN {if ($1) print 0; else print 1}")
        return "$rc"
}

## $1 img1
## $2 img2
## $3 image props (aka -depth 8 -size 16x16), optional
imagemagick_compare() {
        # shellcheck disable=SC2086 # intentional
        PSNR=$(compare -metric PSNR ${3-} "${1?}" "${2?}" null: 2>&1 |
                cut -d\  -f1 || true)
        echo PSNR: "$PSNR (required $REQUESTED_PSNR)"
        # TODO TOREMOVE if not needed (supposing it is IM bug, not feature)
        if expr "$PSNR" : '[0-9][0-9.]*$' >/dev/null &&
                numeric_compare "$PSNR > $BROKEN_IM_THR"; then
                echo "Broken ImageMagick!"
                if ! command -v gm >/dev/null; then
                        return 2
                fi
                echo "will try graphicsmagick instead..."
                PSNR=
        fi
        # IM not found, try GraphicsMagick
        if [ ! "$PSNR" ]; then
                # shellcheck disable=SC2086 # intentional for $3
                PSNR=$(gm compare -metric PSNR ${3-} "${1?}" "${2?}" \
                         | sed -n '/Total: / s/^.*Total: *\([0-9.]*\)/\1/p')
        fi
        if [ "$PSNR" != inf ] && numeric_compare "$PSNR != 0" &&
        numeric_compare "$PSNR < $REQUESTED_PSNR"; then
                return 1
        fi
}

test_commit_b620be2() {
        $GPUJPEG -e -s 1920x1080 -r 1 -f 444-u8-p0p1p2 /dev/zero out.jpg
        $GPUJPEG -d out.jpg out.rgb
        SIZE=$(stat -c %s out.rgb)
        dd if=/dev/zero bs="$SIZE" count=1 of=in.rgb
        if ! imagemagick_compare out.rgb in.rgb "-depth 8 -size 1920x1080"
        then
                echo " black pattern doesn't match!" >&2
                exit 1
        fi

        $GPUJPEG -e -s 16x16 -r 1 -f u8 /dev/zero out.jpg
        $GPUJPEG -d out.jpg out.r
        SIZE=$(stat -c %s out.r)
        dd if=/dev/zero bs="$SIZE" count=1 of=in.r
        if ! imagemagick_compare gray:out.r gray:in.r "-depth 8 -size 16x16"
        then
                echo "grayscale pattern doesn't match!" >&2
                exit 1
        fi

        rm in.rgb in.r out.rgb out.r out.jpg
}

# the single test actually covers both fixes
test_fix_decode_outside_pinned_AND_fix_huff_buf_partially_not_cleared() {
        "$GPUJPEG" -e 392x386.p_u8.noise.tst out.jpg
        "$GPUJPEG" -d out.jpg out.pnm
        if ! imagemagick_compare out.jpg out.pnm
        then
                echo "pattern doesn't match!" >&2
                exit 1
        fi
        rm out.jpg out.pnm
}

# commits e52abeab (increasing size) 791a9e6b (shrinking) crashes
test_different_sizes() {
        param_enc=
        param_dec=
        files=
        for dim in 32 1024 64 2048; do
                s=${dim}x$dim
                printf "P6\n%d %d\n255\n" $dim $dim > $s.pnm
                dd if=/dev/zero of="$s".pnm bs=$((dim*dim*3)) count=1 oflag=append conv=notrunc
                param_enc="$param_enc $s.pnm $s.jpg"
                param_dec="$param_dec $s.jpg $s.pam"
                files="$files $s.pam $s.jpg $s.pnm"
        done

        # shellcheck disable=SC2086 # intentional
        "$GPUJPEG" -e $param_enc
        # shellcheck disable=SC2086 # intentional
        "$GPUJPEG" -d $param_dec
        # shellcheck disable=SC2086 # intentional
        rm $files
}

# sanity test (gpujpeg should fail)
test_nonexistent() {
        ! $GPUJPEG -e nonexistent.pam fail.jpg
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

# test the gray_image.jpg from TwelveMonkeys sample set that caused
# problems (see Git history)
test_gray_image() {
        filename=gray-sample.jpg

        if [ ! -f "$filename" ]; then
                url="https://github.com/haraldk/TwelveMonkeys/blob/master/\
imageio/imageio-jpeg/src/test/resources/jpeg/$filename?raw=true"
                if ! curl -LO "$url"; then
                        echo "Cannot download the image $filename from $url"
                        return
                fi
        fi

        "$GPUJPEG" -d "$filename" out.pnm

        if ! imagemagick_compare "$filename" out.pnm
        then
                echo "$filename doesn't match!" >&2
                exit 1
        fi

        # keeping the $filename intentionally as a cache
        rm out.pnm
}

test_commit_b620be2
test_different_sizes
test_fix_decode_outside_pinned_AND_fix_huff_buf_partially_not_cleared
test_gray_image
test_nonexistent
test_pam_pnm_y4m

