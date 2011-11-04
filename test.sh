#!/bin/bash
#
# Copyright (c) 2011, Martin Srom
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

# Print script help
function print_help {
    echo "Usage:"
    echo "  test.sh name image_size image.rgb";
}

# Return the value of an operation
function float_value() {
     echo | awk 'END { print '"$1"'; }'
}

# Return status code of a comparison
function float_test() {
     echo | awk 'END { exit ( !( '"$1"')); }'
}

# Compute statistic info for array (meidan, avg, min, max)
function compute_statistic() {
    local VALUES=${1}
    # Compute AVG, MIN, MAX
    local COUNT=0
    local SUM=0
    local MIN=99999
    local MAX=0
    local ARRAY=
    for VALUE in ${VALUES}
    do
        ARRAY[$COUNT]=$VALUE
        COUNT=$((${COUNT} + 1))
        SUM=$(float_value "${SUM} + ${VALUE}")
        if `float_test "${VALUE} < ${MIN}"`
        then
            MIN=${VALUE}
        fi
        if `float_test "${VALUE} > ${MAX}"`
        then
            MAX=${VALUE};
        fi
    done
    local AVG=$(float_value "$SUM / $COUNT")
    # Sort values for choosing median
    change=1
    while [ $change -gt 0 ]
    do
        change=0
        for (( i = 0; i < $((${COUNT} - 1)); i++ ))
        do
            if `float_test "${ARRAY[$i]} > ${ARRAY[$(($i + 1))]}"`
            then
                local TEMP=${ARRAY[$i]}
                ARRAY[$i]=${ARRAY[$(($i+1))]}
                ARRAY[$(($i + 1))]=$TEMP
                change=1
            fi
        done
    done
    # Choose median
    MEDIAN=${ARRAY[$((COUNT/2))]}
    echo "${MEDIAN} ${AVG} ${MIN} ${MAX}"
}

# Encode image
function test_encode() {
    local IMAGE_INPUT=${1}
    local IMAGE_OUTPUT=${2}
    local PARAMETERS=${3}
    local IMAGE_COUNT=${4}
    
    local IMAGES=""
    for (( i = 0; i < ${IMAGE_COUNT}; i++ )) 
    do
        IMAGES="${IMAGES} ${IMAGE_INPUT} ${IMAGE_OUTPUT}"
    done
    
    RESULT=$(./jpeg_compress --encode ${PARAMETERS} ${IMAGES} | grep "Encode Image:" | sed "s/Encode Image: *//" | sed "s/ ms//")
    STAT=$(compute_statistic "${RESULT}")
    echo "${STAT}"
}

# Decode image
function test_decode() {
    local IMAGE_INPUT=${1}
    local IMAGE_OUTPUT=${2}
    local PARAMETERS=${3}
    local IMAGE_COUNT=${4}
    
    local IMAGES=""
    for ((i = 0; i < ${IMAGE_COUNT}; i++)) 
    do
        IMAGES="${IMAGES} ${IMAGE_INPUT} ${IMAGE_OUTPUT}"
    done
    
    RESULT=$(./jpeg_compress --decode ${PARAMETERS} ${IMAGES} | grep "Decode Image:" | sed "s/Decode Image: *//" | sed "s/ ms//")
    STAT=$(compute_statistic "${RESULT}")
    echo "${STAT}"
}

# Test encoding and decoding for specified quality
function test() {
    local NAME=${1}
    local IMAGE_SIZE=${2}
    local IMAGE=${3}
    local IMAGE_COUNT=${4}
    local QUALITY=${5}
    
    # Test Encode
    ENCODE_IMAGE_INPUT=${IMAGE}
    ENCODE_IMAGE_OUTPUT=$(echo "${IMAGE}.jpg" | sed "s/.rgb//")
    ENCODE_PARAMETERS="--quality=${QUALITY} --restart=8 --size=${IMAGE_SIZE}"
    ENCODE_RESULT=$(test_encode "${ENCODE_IMAGE_INPUT}" "${ENCODE_IMAGE_OUTPUT}" "${ENCODE_PARAMETERS}" ${IMAGE_COUNT})
    # Compute Encode PNSR
    convert -depth 8 -size ${IMAGE_SIZE} rgb:${ENCODE_IMAGE_INPUT} __original.bmp
    convert ${ENCODE_IMAGE_OUTPUT} __compressed.bmp
    ENCODE_PNSR=$(compare -metric psnr -size ${IMAGE_SIZE} __compressed.bmp __original.bmp __diff.bmp 2>&1)
    rm __original.bmp __compressed.bmp __diff.bmp

    # Test Decode
    DECODE_IMAGE_INPUT=${ENCODE_IMAGE_OUTPUT}
    DECODE_IMAGE_OUTPUT=$(echo "${IMAGE}_decoded.rgb" | sed "s/.rgb//")
    DECODE_PARAMETERS=""
    DECODE_NAME="[${NAME}] [${QUALITY}] [decode]"
    DECODE_RESULT=$(test_decode "${DECODE_IMAGE_INPUT}" "${DECODE_IMAGE_OUTPUT}" "${DECODE_PARAMETERS}" ${IMAGE_COUNT})
    # Compute Decode PNSR
    convert ${DECODE_IMAGE_INPUT} __original.bmp
    convert -depth 8 -size ${IMAGE_SIZE} rgb:${DECODE_IMAGE_OUTPUT} __decompressed.bmp
    DECODE_PNSR=$(compare -metric psnr -size ${IMAGE_SIZE} __decompressed.bmp __original.bmp __diff.bmp 2>&1)
    rm __original.bmp __decompressed.bmp __diff.bmp
    
    echo "[${NAME}] ${QUALITY} ENCODE ${ENCODE_RESULT} ${ENCODE_PNSR} DECODE ${DECODE_RESULT} ${DECODE_PNSR}"
}

# Parse input parameters
NAME=${1}
IMAGE_SIZE=${2}
IMAGE=${3}
IMAGE_COUNT=10

# Check input parameters
if [ "${NAME}" = "" ] || [ "${IMAGE_SIZE}" = "" ] || [ "${IMAGE}" = "" ]
then
    if [ "${NAME}" = "" ]
    then
       echo "Supply test name, for instance 'kepler'!"
    fi
    if [ "${IMAGE_SIZE}" = "" ]
    then
       echo "Supply image size, for instance '1920x1080'!"
    fi
    if [ "${IMAGE}" = "" ]
    then
       echo "Supply image file name, for instance 'image_hd.rgb'!"
    fi
    print_help;
    exit;
fi

echo "[${NAME}] quality ENCODE median avg min max pnsr DECODE median avg min max pnsr"
for (( QUALITY = 100; QUALITY > 0; QUALITY-=5 ))
do
    test ${NAME} ${IMAGE_SIZE} ${IMAGE} ${IMAGE_COUNT} ${QUALITY}
done
