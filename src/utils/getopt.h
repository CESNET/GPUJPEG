/* $OpenBSD: getopt.h,v 1.3 2013/11/22 21:32:49 millert Exp $ */
/* $NetBSD: getopt.h,v 1.4 2000/07/07 10:43:54 ad Exp $ */

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

#ifndef _GETOPT_H_
#define _GETOPT_H_

#include <crtdefs.h>
#include <wchar.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GNU-like getopt_long()
 */
#define no_argument        0
#define required_argument  1
#define optional_argument  2

struct option {
    /* name of long option */
    const char *name;
    /*
     * one of no_argument, required_argument, and optional_argument:
     * whether option takes an argument
     */
    int has_arg;
    /* if not NULL, set *flag to val when option found */
    int *flag;
    /* if flag not NULL, value to set *flag to; else return value */
    int val;
};

struct _woption {
    const wchar_t *name;
    int has_arg;
    int *flag;
    int val;
};


int getopt(int argc, char * const *argv, const char *optstring);
int _wgetopt(int argc, wchar_t * const *argv, const wchar_t *optstring);

int getopt_long(int argc, char * const *argv,
            const char *optstring,
            const struct option *longopts, int *longindex);
int getopt_long_only(int argc, char * const *argv,
            const char *optstring,
            const struct option *longopts, int *longindex);

int _wgetopt_long(int argc, wchar_t * const *argv,
            const wchar_t *optstring,
            const struct _woption *longopts, int *longindex);
int _wgetopt_long_only(int argc, wchar_t * const *argv,
            const wchar_t *optstring,
            const struct _woption *longopts, int *longindex);


/* getopt(3) external variables */
extern char *optarg;       /* argument associated with option */
extern wchar_t *_woptarg;  /* argument associated with option */
extern int opterr;         /* if error message should be printed */
extern int optind;         /* index into parent argv vector */
extern int optopt;         /* character checked for validity */
extern int optreset;       /* reset getopt */


#ifdef _UNICODE
#define _tgetopt            _wgetopt
#define _tgetopt_long       _wgetopt_long
#define _tgetopt_long_only  _wgetopt_long_only
#define _toption            _woption
#define _toptarg            _woptarg
#else
#define _tgetopt            getopt
#define _tgetopt_long       getopt_long
#define _tgetopt_long_only  getopt_long_only
#define _toption            option
#define _toptarg            optarg
#endif

#ifdef __cplusplus
}
#endif

#endif /* !_GETOPT_H_ */
