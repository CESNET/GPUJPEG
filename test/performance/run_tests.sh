#!/bin/sh -eu

DIR=$(dirname "$0")
GPUJPEG=${1:-$DIR/../../gpujpegtool}
if [ ! -x "$GPUJPEG" ] && [ -x "$DIR/../../build/gpujpegtool" ]; then
        GPUJPEG=$DIR/../../build/gpujpegtool
fi

test_4k_synthetic () (
        export VGPU_COUNTERS=1
        res=$(mktemp)
        vgpu run --gpu nvidia/rtx3080ti "$GPUJPEG" -q 75 -e \
                3840x2160.random.tst /dev/null 2>&1 | grep -E \
                '\[counters\]|inst_executed|global  ld' >"$res"
        ref=$(mktemp)
        cat <<EOF >"$ref"
557115 0
779673600 24883200
207521280 16588800
207521280 16588800
207521280 16588800
1458726728 78382080
503568483 2824176
25031600 1486840
EOF
        exec 8<"$res"
        exec 9<"$ref"
        while read -r header <&8; do
                read -r line <&8
                inst_exec=$(printf '%s' "$line" | sed 's/.*inst_executed=\([0-9]*\).*/\1/')
                read -r line <&8
                gl_load=$(printf '%s' "$line" | sed 's/.*global  ld=\([0-9]*\).*/\1/')
                read -r ref_inst_exec ref_gl_load <&9
                inst_perc=$(awk "BEGIN {print int($inst_exec * 100 / $ref_inst_exec) }")
                if [ "$ref_gl_load" -eq 0 ]; then
                        continue # prevent 0-division for first kernel
                fi
                gl_perc=$(awk "BEGIN {print int($gl_load * 100 / $ref_gl_load) }")
                if [ "$inst_perc" -ge 130 ]; then
                        echo "duration $header is $inst_perc% of the reference (>=130%)!" >&2
                        exit 1
                fi
                if [ "$gl_perc" -ge 130 ]; then
                        echo "global access in $header is $gl_perc% of the reference (>=130%)!" >&2
                        exit 1
                fi
                if [ ! "$CI" ]; then
                        continue
                fi
                if [ "$inst_perc" -ge 110 ]; then
                        echo "duration $header is $inst_perc% of the reference (>=110%)!" >>/tmp/warnings
                fi
                if [ "$gl_perc" -ge 110 ]; then
                        echo "global access in $header is $gl_perc% of the reference (>=110%)!" >>/tmp/warnings
                fi
        done
)

tests="
test_4k_synthetic
"

if ! command -v vgpu >/dev/null; then
        echo "PantheonSim not found (vgpu not in PATH) - skipping tests" >&1
        if [ "$CI" ]; then
                exit 1
        fi
        exit 100
fi

for n in $tests; do
        printf "\n\n\nRunning %s...\n" "$n"
        "$n"
done
