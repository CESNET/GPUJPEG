#!/bin/sh
srcdir=$(dirname "$0")
test -z "$srcdir" && srcdir=.

ORIGDIR=$(pwd)

cd "$srcdir" || return

uname_s=$(uname -s)
if [ "$uname_s" = "Darwin" ]; then
    LIBTOOLIZE=glibtoolize
else 
    LIBTOOLIZE=libtoolize
fi

$LIBTOOLIZE --copy && \
( [ -d m4 ] || mkdir m4 ) && \
aclocal -I m4 && \
automake --copy --add-missing && \
autoconf && \
cd "$ORIGDIR" && "$srcdir/configure" "$@"

STATUS=$?

([ $STATUS -eq 0 ] && echo "Autogen done." ) || echo "Autogen failed."

echo "Note: CMake is now preferred for GPUJPEG build configuration."

if expr "$(uname -s)" : MSYS >/dev/null; then
    printf "\nIt is highly probable that autoconf won't work in MSW - \
use cmake if possible\n"
fi

exit $STATUS

