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

function print_help
{
    echo "Usage:"
    echo "  test.sh name image_size image.rgb";
}

function encode()
{
    local IMAGE_INPUT=${1}
    local IMAGE_OUTPUT=${2}
    local PARAMETERS=${3}
    local IMAGE_COUNT=${4}
    
    echo "[${NAME}] Encoding image [${IMAGE_INPUT}] to [${IMAGE_OUTPUT}]"
    
    local IMAGES=""
    for ((i = 0; i < ${IMAGE_COUNT}; i++)) 
    do
        IMAGES="${IMAGES} ${IMAGE_INPUT} ${IMAGE_OUTPUT}"
    done
    
    RESULT=$(./jpeg_compress --encode ${PARAMETERS} ${IMAGES} | grep "Encode Image:" | sed "s/Encode Image: *//" | sed "s/ ms//")
    for DURATION in $RESULT
    do
        echo "$DURATION"
    done
}

function decode()
{
    local IMAGE_INPUT=${1}
    local IMAGE_OUTPUT=${2}
    local PARAMETERS=${3}
    local IMAGE_COUNT=${4}
    
    echo "[${NAME}] Decoding image [${IMAGE_INPUT}] to [${IMAGE_OUTPUT}]"
    
    local IMAGES=""
    for ((i = 0; i < ${IMAGE_COUNT}; i++)) 
    do
        IMAGES="${IMAGES} ${IMAGE_INPUT} ${IMAGE_OUTPUT}"
    done
    
    RESULT=$(./jpeg_compress --decode ${PARAMETERS} ${IMAGES} | grep "Decode Image:" | sed "s/Decode Image: *//" | sed "s/ ms//")
    for DURATION in $RESULT
    do
        echo "$DURATION"
    done
}

# Parse input parameters
NAME=${1}
IMAGE_SIZE=${2}
IMAGE=${3}
RUN_COUNT=10
RUN_IMAGE_COUNT=10

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

# Encode
ENCODE_IMAGE_INPUT=${IMAGE}
ENCODE_IMAGE_OUTPUT=$(echo "${IMAGE}.jpg" | sed "s/.rgb//")
ENCODE_PARAMETERS="--quality=90 --restart=8 --size=${IMAGE_SIZE}"
encode "${ENCODE_IMAGE_INPUT}" "${ENCODE_IMAGE_OUTPUT}" "${ENCODE_PARAMETERS}" ${RUN_IMAGE_COUNT}

# Encode
DECODE_IMAGE_INPUT=${ENCODE_IMAGE_OUTPUT}
DECODE_IMAGE_OUTPUT=$(echo "${IMAGE}_decoded.rgb" | sed "s/.rgb//")
DECODE_PARAMETERS=""
decode "${DECODE_IMAGE_INPUT}" "${DECODE_IMAGE_OUTPUT}" "${DECODE_PARAMETERS}" ${RUN_IMAGE_COUNT}
