#!/bin/sh -eu

DIR=$(dirname $0)
GPUJPEG=${1:-$DIR/../../gpujpegtool}
if [ ! -x "$GPUJPEG" ] && [ -x "$DIR/../../build/gpujpegtool" ]; then
        GPUJPEG=$DIR/../../build/gpujpegtool
fi

. "$DIR/../common.sh" # for magick_compare

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

tests="
test_random_psnr
"

for n in $tests; do
        printf "\n\n\nRunning %s...\n" "$n"
        "$n"
done
