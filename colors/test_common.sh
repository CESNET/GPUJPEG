#!/bin/bash -eu
#
# Requires FFmpeg (convert), ImageMagick (compare)
# Parameters
# 1         = (optional) path to GPUJPEG executable
# NAME      = image name, e.g. "image_yuv_422"
# EXTENSION = image extension, e.g. "yuv"
# MODE      = image arguments for gpujpeg, e.g. "--colorspace=ycbcr-bt601 --pixel-format=444-u8-p012"
# FF_FORMAT = respective FFmpeg format (until GPUJPEG is capable to convert by itself), eg. "yuv444p"
#
# @todo
# REQUESTED_PSNR should perhaps not be fixed but given by user to match expected value

# Get script folder
DIR=`dirname $0`
GPUJPEG=${1:-$DIR/../gpujpeg}
REQUESTED_PSNR=40
IMAGE=image_bt709_422.yuv
#IMAGE=camera_bt709_422.yuv

if ! command -v compare >/dev/null; then
        echo "compare from ImageMagick not found!" >&2
        exit 2
fi

# Create an image from source in specified mode ()
#$GPUJPEG --size=1920x1080 --colorspace=ycbcr-bt709 --pixel-format=422-u8-p1020 \
#    --convert $MODE $DIR/$IMAGE $DIR/$NAME.$EXTENSION
ffmpeg -y -f rawvideo -pixel_format uyvy422 -video_size 1920x1080 -i $DIR/$IMAGE -f rawvideo -pix_fmt $FF_FORMAT -video_size 1920x1080 $DIR/$NAME.$EXTENSION

# Encode and Decode the image
$GPUJPEG --size 1920x1080 $MODE \
    --encode --quality 100 $DIR/$NAME.$EXTENSION $DIR/$NAME.encoded.jpg
$GPUJPEG $MODE \
    --decode $DIR/$NAME.encoded.jpg $DIR/$NAME.decoded.$EXTENSION

# Convert the Original and the Processed Image to RGB444
#$GPUJPEG --size=1920x1080 $MODE \
#    --convert --colorspace=rgb $DIR/$NAME.$EXTENSION $DIR/$NAME.rgb
#$GPUJPEG --size=1920x1080 $MODE \
#    --convert --colorspace=rgb $DIR/$NAME.decoded.$EXTENSION $DIR/$NAME.decoded.rgb
if [ $EXTENSION != rgb ]; then
        ffmpeg -y -f rawvideo -pixel_format $FF_FORMAT -video_size 1920x1080 -i  $DIR/$NAME.$EXTENSION -f rawvideo -pix_fmt rgb24 -video_size 1920x1080 $DIR/$NAME.rgb
        ffmpeg -y -f rawvideo -pixel_format $FF_FORMAT -video_size 1920x1080 -i  $DIR/$NAME.decoded.$EXTENSION -f rawvideo -pix_fmt rgb24 -video_size 1920x1080 $DIR/$NAME.decoded.rgb
fi

# Display Left/Right Diff of the Original and the Processed Image
#$DIR/display_diff.sh $DIR/$NAME.rgb $DIR/$NAME.decoded.rgb
PSNR=`compare -metric PSNR -depth 8 -size 1920x1080  $DIR/$NAME.rgb $DIR/$NAME.decoded.rgb null: 2>&1 || true`

echo PSNR: $PSNR

if expr $PSNR \< $REQUESTED_PSNR; then
        exit 1
fi

# Delete Created Files
rm -f $DIR/$NAME.$EXTENSION $DIR/$NAME.rgb $DIR/$NAME.encoded.jpg $DIR/$NAME.decoded.$EXTENSION $DIR/$NAME.decoded.rgb

