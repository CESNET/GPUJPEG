/* $OpenBSD: getopt_long.c,v 1.32 2020/05/27 22:25:09 schwarze Exp $ */
/* $NetBSD: getopt_long.c,v 1.15 2002/01/31 22:43:40 tv Exp $ */

/*
 * Copyright (c) 2002 Todd C. Miller <millert@openbsd.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * Sponsored in part by the Defense Advanced Research Projects
 * Agency (DARPA) and Air Force Research Laboratory, Air Force
 * Materiel Command, USAF, under agreement number F39502-99-1-0512.
 */
/*-
 * Copyright (c) 2000 The NetBSD Foundation, Inc.
 * All rights reserved.
 *
 * This code is derived from software contributed to The NetBSD Foundation
 * by Dieter Baron and Thomas Klausner.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE NETBSD FOUNDATION, INC. AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>
#include <wchar.h>
#include <windows.h>
#include "getopt.h"

int    opterr = 1;        /* if error message should be printed */
int    optind = 1;        /* index into parent argv vector */
int    optopt = _T('?');  /* character checked for validity */
int    optreset;          /* reset getopt */
TCHAR *_toptarg;          /* argument associated with option */

#define PRINT_ERROR    ((opterr) && (*options != _T(':')))

#define FLAG_PERMUTE   0x01  /* permute non-options to the end of argv */
#define FLAG_ALLARGS   0x02  /* treat non-options as args to option "-1" */
#define FLAG_LONGONLY  0x04  /* operate as getopt_long_only */

/* return values */
#define BADCH    (int)_T('?')
#define BADARG   ((*options == _T(':')) ? (int)_T(':') : BADCH)
#define INORDER  (int)1

#define EMSG     _T("")

static int getopt_internal(int, TCHAR * const *, const TCHAR *,
                const struct _toption *, int *, int);
static int parse_long_options(TCHAR * const *, const TCHAR *,
                const struct _toption *, int *, int, int);
static int gcd(int, int);
static void permute_args(int, int, int, TCHAR * const *);
static void xwarnx(const TCHAR *fmt, ...);
static int envset(const char *name);

static TCHAR *place = EMSG; /* option letter processing */

/* XXX: set optreset to 1 rather than these two */
static int nonopt_start = -1; /* first non option argument (for permute) */
static int nonopt_end = -1;   /* first option after non options (for permute) */

/* Error messages */
static const TCHAR recargchar[]   = _T("option requires an argument -- %c");
static const TCHAR recargstring[] = _T("option requires an argument -- %s");
static const TCHAR ambig[]        = _T("ambiguous option -- %.*s");
static const TCHAR noarg[]        = _T("option doesn't take an argument -- %.*s");
static const TCHAR illoptchar[]   = _T("unknown option -- %c");
static const TCHAR illoptstring[] = _T("unknown option -- %s");

static void
xwarnx(const TCHAR *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    if (__targv[0] != NULL)
        (void)_ftprintf(stderr, _T("%s: "), __targv[0]);
    if (fmt != NULL)
        (void)_vftprintf(stderr, fmt, ap);
    (void)_ftprintf(stderr, _T("\n"));
    va_end(ap);
}

static int
envset(const char *name)
{
#ifdef _MSC_VER
    size_t len = 0;

    getenv_s(&len, NULL, 0, name);

    return (len == 0) ? -1 : 0;
#else
    return (getenv(name) == NULL) ? -1 : 0;
#endif
}

/*
 * Compute the greatest common divisor of a and b.
 */
static int
gcd(int a, int b)
{
    int c;

    c = a % b;
    while (c != 0) {
        a = b;
        b = c;
        c = a % b;
    }

    return (b);
}

/*
 * Exchange the block from nonopt_start to nonopt_end with the block
 * from nonopt_end to opt_end (keeping the same order of arguments
 * in each block).
 */
static void
permute_args(int panonopt_start, int panonopt_end, int opt_end,
    TCHAR * const *nargv)
{
    int cstart, cyclelen, i, j, ncycle, nnonopts, nopts, pos;
    TCHAR *swap;

    /*
     * compute lengths of blocks and number and size of cycles
     */
    nnonopts = panonopt_end - panonopt_start;
    nopts = opt_end - panonopt_end;
    ncycle = gcd(nnonopts, nopts);
    cyclelen = (opt_end - panonopt_start) / ncycle;

    for (i = 0; i < ncycle; i++) {
        cstart = panonopt_end+i;
        pos = cstart;
        for (j = 0; j < cyclelen; j++) {
            if (pos >= panonopt_end)
                pos -= nnonopts;
            else
                pos += nopts;
            swap = nargv[pos];
            ((TCHAR **)nargv)[pos] = nargv[cstart];
            ((TCHAR **)nargv)[cstart] = swap;
        }
    }
}

/*
 * parse_long_options --
 *    Parse long options in argc/argv argument vector.
 * Returns -1 if short_too is set and the option does not match long_options.
 */
static int
parse_long_options(TCHAR * const *nargv, const TCHAR *options,
    const struct _toption *long_options, int *idx, int short_too, int flags)
{
    TCHAR *current_argv, *has_equal;
    size_t current_argv_len;
    int i, match, exact_match, second_partial_match;

    current_argv = place;
    match = -1;
    exact_match = 0;
    second_partial_match = 0;

    optind++;

    if ((has_equal = _tcschr(current_argv, _T('='))) != NULL) {
        /* argument found (--option=arg) */
        current_argv_len = has_equal - current_argv;
        has_equal++;
    } else
        current_argv_len = _tcslen(current_argv);

    for (i = 0; long_options[i].name; i++) {
        /* find matching long option */
        if (_tcsncmp(current_argv, long_options[i].name,
            current_argv_len))
            continue;

        if (_tcslen(long_options[i].name) == current_argv_len) {
            /* exact match */
            match = i;
            exact_match = 1;
            break;
        }
        /*
         * If this is a known short option, don't allow
         * a partial match of a single character.
         */
        if (short_too && current_argv_len == 1)
            continue;

        if (match == -1)    /* first partial match */
            match = i;
        else if ((flags & FLAG_LONGONLY) ||
            long_options[i].has_arg != long_options[match].has_arg ||
            long_options[i].flag != long_options[match].flag ||
            long_options[i].val != long_options[match].val)
            second_partial_match = 1;
    }
    if (!exact_match && second_partial_match) {
        /* ambiguous abbreviation */
        if (PRINT_ERROR)
            xwarnx(ambig, (int)current_argv_len,
                current_argv);
        optopt = 0;
        return (BADCH);
    }
    if (match != -1) {        /* option found */
        if (long_options[match].has_arg == no_argument
            && has_equal) {
            if (PRINT_ERROR)
                xwarnx(noarg, (int)current_argv_len,
                    current_argv);
            /*
             * XXX: GNU sets optopt to val regardless of flag
             */
            if (long_options[match].flag == NULL)
                optopt = long_options[match].val;
            else
                optopt = 0;
            return (BADARG);
        }
        if (long_options[match].has_arg == required_argument ||
            long_options[match].has_arg == optional_argument) {
            if (has_equal)
                _toptarg = has_equal;
            else if (long_options[match].has_arg ==
                required_argument) {
                /*
                 * optional argument doesn't use next nargv
                 */
                _toptarg = nargv[optind++];
            }
        }
        if ((long_options[match].has_arg == required_argument)
            && (_toptarg == NULL)) {
            /*
             * Missing argument; leading ':' indicates no error
             * should be generated.
             */
            if (PRINT_ERROR)
                xwarnx(recargstring, current_argv);
            /*
             * XXX: GNU sets optopt to val regardless of flag
             */
            if (long_options[match].flag == NULL)
                optopt = long_options[match].val;
            else
                optopt = 0;
            --optind;
            return (BADARG);
        }
    } else {            /* unknown option */
        if (short_too) {
            --optind;
            return (-1);
        }
        if (PRINT_ERROR)
            xwarnx(illoptstring, current_argv);
        optopt = 0;
        return (BADCH);
    }
    if (idx)
        *idx = match;
    if (long_options[match].flag) {
        *long_options[match].flag = long_options[match].val;
        return (0);
    } else
        return (long_options[match].val);
}

/*
 * getopt_internal --
 *    Parse argc/argv argument vector.  Called by user level routines.
 */
static int
getopt_internal(int nargc, TCHAR * const *nargv, const TCHAR *options,
    const struct _toption *long_options, int *idx, int flags)
{
    TCHAR *oli;                /* option letter list index */
    int optchar, short_too;
    static int posixly_correct = -1;

    if (options == NULL)
        return (-1);

    /*
     * XXX Some GNU programs (like cvs) set optind to 0 instead of
     * XXX using optreset.  Work around this braindamage.
     */
    if (optind == 0)
        optind = optreset = 1;

    /*
     * Disable GNU extensions if POSIXLY_CORRECT is set or options
     * string begins with a '+'.
     */
    if (posixly_correct == -1 || optreset)
        posixly_correct = envset("POSIXLY_CORRECT");
    if (*options == _T('-'))
        flags |= FLAG_ALLARGS;
    else if (posixly_correct != -1 || *options == _T('+'))
        flags &= ~FLAG_PERMUTE;
    if (*options == _T('+') || *options == _T('-'))
        options++;

    _toptarg = NULL;
    if (optreset)
        nonopt_start = nonopt_end = -1;
start:
    if (optreset || !*place) {        /* update scanning pointer */
        optreset = 0;
        if (optind >= nargc) {          /* end of argument vector */
            place = EMSG;
            if (nonopt_end != -1) {
                /* do permutation, if we have to */
                permute_args(nonopt_start, nonopt_end,
                    optind, nargv);
                optind -= nonopt_end - nonopt_start;
            }
            else if (nonopt_start != -1) {
                /*
                 * If we skipped non-options, set optind
                 * to the first of them.
                 */
                optind = nonopt_start;
            }
            nonopt_start = nonopt_end = -1;
            return (-1);
        }
        if (*(place = nargv[optind]) != _T('-') ||
            (place[1] == 0 && _tcschr(options, _T('-')) == NULL)) {
            place = EMSG;        /* found non-option */
            if (flags & FLAG_ALLARGS) {
                /*
                 * GNU extension:
                 * return non-option as argument to option 1
                 */
                _toptarg = nargv[optind++];
                return (INORDER);
            }
            if (!(flags & FLAG_PERMUTE)) {
                /*
                 * If no permutation wanted, stop parsing
                 * at first non-option.
                 */
                return (-1);
            }
            /* do permutation */
            if (nonopt_start == -1)
                nonopt_start = optind;
            else if (nonopt_end != -1) {
                permute_args(nonopt_start, nonopt_end,
                    optind, nargv);
                nonopt_start = optind -
                    (nonopt_end - nonopt_start);
                nonopt_end = -1;
            }
            optind++;
            /* process next argument */
            goto start;
        }
        if (nonopt_start != -1 && nonopt_end == -1)
            nonopt_end = optind;

        /*
         * If we have "-" do nothing, if "--" we are done.
         */
        if (place[1] != 0 && *++place == _T('-') && place[1] == 0) {
            optind++;
            place = EMSG;
            /*
             * We found an option (--), so if we skipped
             * non-options, we have to permute.
             */
            if (nonopt_end != -1) {
                permute_args(nonopt_start, nonopt_end,
                    optind, nargv);
                optind -= nonopt_end - nonopt_start;
            }
            nonopt_start = nonopt_end = -1;
            return (-1);
        }
    }

    /*
     * Check long options if:
     *  1) we were passed some
     *  2) the arg is not just "-"
     *  3) either the arg starts with -- we are getopt_long_only()
     */
    if (long_options != NULL && place != nargv[optind] &&
        (*place == _T('-') || (flags & FLAG_LONGONLY))) {
        short_too = 0;
        if (*place == _T('-'))
            place++;        /* --foo long option */
        else if (*place != _T(':') && _tcschr(options, *place) != NULL)
            short_too = 1;        /* could be short option too */

        optchar = parse_long_options(nargv, options, long_options,
            idx, short_too, flags);
        if (optchar != -1) {
            place = EMSG;
            return (optchar);
        }
    }

    if ((optchar = (int)*place++) == (int)_T(':') ||
        (oli = _tcschr(options, optchar)) == NULL) {
        if (!*place)
            ++optind;
        if (PRINT_ERROR)
            xwarnx(illoptchar, optchar);
        optopt = optchar;
        return (BADCH);
    }
    if (long_options != NULL && optchar == _T('W') && oli[1] == _T(';')) {
        /* -W long-option */
        if (*place)            /* no space */
            /* NOTHING */;
        else if (++optind >= nargc) {    /* no arg */
            place = EMSG;
            if (PRINT_ERROR)
                xwarnx(recargchar, optchar);
            optopt = optchar;
            return (BADARG);
        } else                /* white space */
            place = nargv[optind];
        optchar = parse_long_options(nargv, options, long_options,
            idx, 0, flags);
        place = EMSG;
        return (optchar);
    }
    if (*++oli != _T(':')) {            /* doesn't take argument */
        if (!*place)
            ++optind;
    } else {                /* takes (optional) argument */
        _toptarg = NULL;
        if (*place)            /* no white space */
            _toptarg = place;
        else if (oli[1] != _T(':')) {    /* arg not optional */
            if (++optind >= nargc) {    /* no arg */
                place = EMSG;
                if (PRINT_ERROR)
                    xwarnx(recargchar, optchar);
                optopt = optchar;
                return (BADARG);
            } else
                _toptarg = nargv[optind];
        }
        place = EMSG;
        ++optind;
    }
    /* dump back option letter */
    return (optchar);
}

/*
 * getopt --
 *    Parse argc/argv argument vector.
 */
int
_tgetopt(int nargc, TCHAR * const *nargv, const TCHAR *options)
{
    /*
     * We don't pass FLAG_PERMUTE to getopt_internal() since
     * the BSD getopt(3) (unlike GNU) has never done this.
     *
     * Furthermore, since many privileged programs call getopt()
     * before dropping privileges it makes sense to keep things
     * as simple (and bug-free) as possible.
     */
    return (getopt_internal(nargc, nargv, options, NULL, NULL, 0));
}

/*
 * getopt_long --
 *    Parse argc/argv argument vector.
 */
int
_tgetopt_long(int nargc, TCHAR * const *nargv, const TCHAR *options,
    const struct _toption *long_options, int *idx)
{
    return (getopt_internal(nargc, nargv, options, long_options, idx,
        FLAG_PERMUTE));
}

/*
 * getopt_long_only --
 *    Parse argc/argv argument vector.
 */
int
_tgetopt_long_only(int nargc, TCHAR * const *nargv, const TCHAR *options,
    const struct _toption *long_options, int *idx)
{
    return (getopt_internal(nargc, nargv, options, long_options, idx,
        FLAG_PERMUTE|FLAG_LONGONLY));
}
