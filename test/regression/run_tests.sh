#!/bin/sh -eu

DIR=$(dirname $0)
GPUJPEG=${1:-$DIR/../../gpujpegtool}
if [ ! -x "$GPUJPEG" ] && [ -x "$DIR/../../build/gpujpegtool" ]; then
        GPUJPEG=$DIR/../../build/gpujpegtool
fi

. "$DIR/../common.sh" # for magick_compare

test_commit_b620be2() {
        $GPUJPEG -e -s 1920x1080 -r 1 -f 444-u8-p0p1p2 /dev/zero out.jpg
        $GPUJPEG -d out.jpg out.rgb
        SIZE=$(stat -c %s out.rgb)
        dd if=/dev/zero bs="$SIZE" count=1 of=in.rgb
        magick_compare out.rgb in.rgb "-depth 8 -size 1920x1080"

        $GPUJPEG -e -s 16x16 -r 1 -f u8 /dev/zero out.jpg
        $GPUJPEG -d out.jpg out.r
        SIZE=$(stat -c %s out.r)
        dd if=/dev/zero bs="$SIZE" count=1 of=in.r
        magick_compare gray:out.r gray:in.r "-depth 8 -size 16x16"

        rm in.rgb in.r out.rgb out.r out.jpg
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

# the single test actually covers both fixes
test_fix_decode_outside_pinned_AND_fix_huff_buf_partially_not_cleared() {
        "$GPUJPEG" -e 392x386.p_u8.noise.tst out.jpg
        "$GPUJPEG" -d out.jpg out.pnm
        magick_compare out.jpg out.pnm
        rm out.jpg out.pnm
}

test_fix_postprocess_memcpy_pitch_20250305() {
        $GPUJPEG -e 1119x561.c_ycbcr-jpeg.p_422-u8-p0p1p2.tst ycbcr422.jpg
        $GPUJPEG -d -c ycbcr-jpeg ycbcr422.jpg out.y4m
        rm ycbcr422.jpg out.y4m
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
        magick_compare "$filename" out.pnm

        # keeping the $filename intentionally as a cache
        rm out.pnm
}

# sanity test (gpujpeg should fail)
test_nonexistent() {
        ! $GPUJPEG -e nonexistent.pam fail.jpg
}

# currently just a simple read/write tests without validating file contents
test_pam_pnm_y4m() {
        w=256
        h=256
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

## This test ensures given compression quality preserved
## measured with PSNR. The required PSNR values are set
## to the currently received values (measured with GM 1.3.45)
test_random_psnr() {
        # use weird resolution to catch more errors
        w=$((7*8*20-1))
        h=$((33*17))
        out_patt=91af70a4 # random pattern

        file1=${w}x$h.random.c_rgb
        q1=75
        flags1=
        req_psnr1=22 # actually 22.26 at the time of writing

        file2=${w}x$h.p_u8.random
        q2=75
        flags2=
        req_psnr2=28.4 # actually 28.53

        file3=${w}x$h.p_4444-u8-p0123.random
        q3=90
        flags3='-a -N'
        req_psnr3=36.3 # actually 36.4

        n=1
        while [ "$(eval echo "\$file$n")" ]; do
                eval name="\$file$n"
                eval req_psnr="\$req_psnr$n"
                eval q="\$q$n"
                eval flags="\$flags$n"
                out_n=$out_patt-$name
                "$GPUJPEG" $flags -b -q "$q" -e -s ${w}x$h "$name.tst" in.jpg
                "$GPUJPEG" $flags -d in.jpg "$out_n.XXX"
                echo magick_compare "input-$name".* "$out_n".* '' "$req_psnr"
                magick_compare "input-$name".* "$out_n".* '' "$req_psnr"
                rm "input-$name".* "$out_n".* in.jpg
                n=$((n+1))
        done
}

test_commit_b620be2
test_different_sizes
test_fix_decode_outside_pinned_AND_fix_huff_buf_partially_not_cleared
test_fix_postprocess_memcpy_pitch_20250305
test_gray_image
test_nonexistent
test_pam_pnm_y4m
test_random_psnr

