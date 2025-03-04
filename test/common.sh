numeric_compare() {
        rc=$(awk "BEGIN {if ($1) print 0; else print 1}")
        return "$rc"
}

## $1 img1
## $2 img2
## $3 image props like `-depth 8 -size 16x16` - needed for raw pixwl data,
##    otherwise unneeded
## $4 requested PSNR (optional, if unset $requested_psnr_dfl is used)
magick_compare() {
        requested_psnr_dfl=50
        broken_im_thr=110

        requested_psnr=${4-$requested_psnr_dfl}
        # shellcheck disable=SC2086 # intentional
        psnr=$(compare -metric PSNR ${3-} "${1?}" "${2?}" null: 2>&1 |
                cut -d\  -f1 || true)
        echo PSNR: "$psnr (required $requested_psnr)"
        # TODO TOREMOVE if not needed (supposing it is IM bug, not feature)
        if expr "$psnr" : '[0-9][0-9.]*$' >/dev/null &&
                numeric_compare "$psnr > $broken_im_thr"; then
                echo "Broken ImageMagick!"
                if ! command -v gm >/dev/null; then
                        return 2
                fi
                echo "will try graphicsmagick instead..."
                psnr=
        fi
        # IM not found, try GraphicsMagick
        if [ ! "$psnr" ]; then
                # shellcheck disable=SC2086 # intentional for $3
                psnr=$(gm compare -metric PSNR ${3-} "${1?}" "${2?}" \
                         | sed -n '/Total: / s/^.*Total: *\([0-9.]*\)/\1/p')
        fi
        if [ "$psnr" != inf ] && numeric_compare "$psnr != 0" &&
        numeric_compare "$psnr < $requested_psnr"; then
                return 1
        fi
}

# compat ("image" prefix dropped because GraphicsMagic is also used if
# IM not found /and may be prefered later/)
imagemagick_compare() {
        magick_compare "$@"
}
