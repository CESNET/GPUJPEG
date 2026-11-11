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

readonly subs_in=hdr.ppm
readonly subs_out=gj_test_qual_subs.jpg
test_subsampling_enc() (
        if [ ! -f $subs_in ]; then
                # https://imagecompression.info/test_images/
                url="https://owncloud.cesnet.cz/index.php/s/sFfxE3sRtkYMKhD/download?path=%2F&files=hdr.ppm"
                if ! curl -Lf "$url" -o $subs_in; then
                        echo "Cannot download the image $subs_in from $url" 1>&2
                        if [ "$CI" ]; then
                                exit 1
                        fi
                        return
                fi
        fi

        "$GPUJPEG" -e -S420 -q 95 $subs_in $subs_out
        magick_compare $subs_in $subs_out '' 43.6 # we have PSNR 43.81

        # keeping the $in intentionally as a cache
        # keeping the $out intentionally for test_subsampling_dec
)
# depends on test_subsampling_enc output
test_subsampling_dec() (
        in=$subs_out
        out=gj_test_subs_dec_out.ppm

        if [ ! -f $in ]; then
                return
        fi

        "$GPUJPEG" -d $in $out
        magick_compare $in $out '' 59.4 # we have PSNR 59.60
        # compare to the original
        magick_compare $out $subs_in '' 43.6 # we have PSNR 43.81

        rm $in
        rm $out
)

tests="
test_random_psnr
test_subsampling_enc
test_subsampling_dec
"

for n in $tests; do
        printf "\n\n\nRunning %s...\n" "$n"
        "$n"
done
