/*
 *
 *  This file is part of MUMPS 5.8.0, released
 *  on Tue May  6 08:27:40 UTC 2025
 *
 *
 *  Copyright 1991-2025 CERFACS, CNRS, ENS Lyon, INP Toulouse, Inria,
 *  Mumps Technologies, University of Bordeaux.
 *
 *  This version of MUMPS is provided to you free of charge. It is
 *  released under the CeCILL-C license
 *  (see doc/CeCILL-C_V1-en.txt, doc/CeCILL-C_V1-fr.txt, and
 *  https://cecill.info/licences/Licence_CeCILL-C_V1-en.html)
 *
 */
#include <stdio.h>
#include <stdint.h>
#if defined(USE_AVX512_VBMI)
#include <immintrin.h>
#endif
// NB : mumps_flytes undef __AVX512{F__/VBMI__} flags if USE_AVX512_VBMI is not defined
#include "mumps_flytes.h"
/* this implementation exists to avoid depending on a c++ compiler
 * this is inspired from
 * https://gitlab.com/AntJego/adaptative-precision-blr */
void MUMPS_CALL mumps_flyte_return() {};
