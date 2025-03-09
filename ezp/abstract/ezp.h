/*******************************************************************************
 * Copyright (C) 2025 Theodore Chang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

#ifndef EZP_H
#define EZP_H

#include <complex>

using complex8 = std::complex<float>;
using complex16 = std::complex<double>;

#define EZP_UNDERSCORE

#ifdef EZP_UNDERSCORE
#define EZP_APPEND_UNDERSCORE(name) name##_
#else
#define EZP_APPEND_UNDERSCORE(name) name
#endif

#define EZP(name) EZP_APPEND_UNDERSCORE(name)

#define numroc EZP(numroc)

#define blacs_exit EZP(blacs_exit)
#define blacs_get EZP(blacs_get)
#define blacs_gridexit EZP(blacs_gridexit)
#define blacs_gridinfo EZP(blacs_gridinfo)
#define blacs_gridinit EZP(blacs_gridinit)
#define blacs_pinfo EZP(blacs_pinfo)

#define descinit EZP(descinit)

#define pcgbtrf EZP(pcgbtrf)
#define pcgbtrs EZP(pcgbtrs)
#define pcgemr2d EZP(pcgemr2d)
#define pddbtrf EZP(pddbtrf)
#define pddbtrs EZP(pddbtrs)
#define pdgbsv EZP(pdgbsv)
#define pdgbtrf EZP(pdgbtrf)
#define pdgbtrs EZP(pdgbtrs)
#define pdgemr2d EZP(pdgemr2d)
#define pdgesv EZP(pdgesv)
#define pdgetrf EZP(pdgetrf)
#define pdgetrs EZP(pdgetrs)
#define pdpbtrf EZP(pdpbtrf)
#define pdpbtrs EZP(pdpbtrs)
#define pdposv EZP(pdposv)
#define pdpotrf EZP(pdpotrf)
#define pdpotrs EZP(pdpotrs)
#define psdbtrf EZP(psdbtrf)
#define psdbtrs EZP(psdbtrs)
#define psgbsv EZP(psgbsv)
#define psgbtrf EZP(psgbtrf)
#define psgbtrs EZP(psgbtrs)
#define psgemr2d EZP(psgemr2d)
#define psgesv EZP(psgesv)
#define psgetrf EZP(psgetrf)
#define psgetrs EZP(psgetrs)
#define pspbtrf EZP(pspbtrf)
#define pspbtrs EZP(pspbtrs)
#define psposv EZP(psposv)
#define pspotrf EZP(pspotrf)
#define pspotrs EZP(pspotrs)
#define pzgbtrf EZP(pzgbtrf)
#define pzgbtrs EZP(pzgbtrs)
#define pzgemr2d EZP(pzgemr2d)

#define igamn2d EZP(igamn2d)
#define igamx2d EZP(igamx2d)
#define igebr2d EZP(igebr2d)

#ifdef __cplusplus
extern "C" {
#endif

int numroc(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);

void blacs_exit(const int* notDone);
void blacs_get(const int* ConTxt, const int* what, int* val);
void blacs_gridexit(const int* ConTxt);
void blacs_gridinfo(const int* ConTxt, int* nprow, int* npcol, int* myrow, int* mycol);
void blacs_gridinit(int* ConTxt, const char* layout, const int* nprow, const int* npcol);
void blacs_pinfo(int* mypnum, int* nprocs);

void descinit(int* desc, const int* m, const int* n, const int* mb, const int* nb, const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

void pcgbtrf(const int* n, const int* bwl, const int* bwu, complex8* a, const int* ja, const int* desca, int* ipiv, complex8* af, const int* laf, complex8* work, const int* lwork, int* info);
void pcgbtrs(const char* trans, const int* n, const int* bwl, const int* bwu, const int* nrhs, complex8* a, const int* ja, const int* desca, int* ipiv, complex8* b, const int* ib, const int* descb, complex8* af, const int* laf, complex8* work, const int* lwork, int* info);
void pcgemr2d(const int* m, const int* n, const complex8* a, const int* ia, const int* ja, const int* desca, complex8* b, const int* ib, const int* jb, const int* descb, const int* ictxt);
void pddbtrf(const int* n, const int* bwl, const int* bwu, double* a, const int* ja, const int* desca, double* af, const int* laf, double* work, const int* lwork, int* info);
void pddbtrs(const char* trans, const int* n, const int* bwl, const int* bwu, const int* nrhs, double* a, const int* ja, const int* desca, double* b, const int* ib, const int* descb, double* af, const int* laf, double* work, const int* lwork, int* info);
void pdgbsv(const int* n, const int* bwl, const int* bwu, const int* nrhs, double* a, const int* ja, const int* desca, int* ipiv, double* b, const int* ib, const int* descb, double* work, const int* lwork, int* info);
void pdgbtrf(const int* n, const int* bwl, const int* bwu, double* a, const int* ja, const int* desca, int* ipiv, double* af, const int* laf, double* work, const int* lwork, int* info);
void pdgbtrs(const char* trans, const int* n, const int* bwl, const int* bwu, const int* nrhs, double* a, const int* ja, const int* desca, int* ipiv, double* b, const int* ib, const int* descb, double* af, const int* laf, double* work, const int* lwork, int* info);
void pdgemr2d(const int* m, const int* n, const double* a, const int* ia, const int* ja, const int* desca, double* b, const int* ib, const int* jb, const int* descb, const int* ictxt);
void pdgesv(const int* n, const int* nrhs, double* a, const int* ia, const int* ja, const int* desca, int* ipiv, double* b, const int* ib, const int* jb, const int* descb, int* info);
void pdgetrf(const int* m, const int* n, double* a, const int* ia, const int* ja, const int* desca, int* ipiv, int* info);
void pdgetrs(const char* trans, const int* n, const int* nrhs, const double* a, const int* ia, const int* ja, const int* desca, const int* ipiv, double* b, const int* ib, const int* jb, const int* descb, int* info);
void pdpbtrf(const char* uplo, const int* n, const int* bw, double* a, const int* ja, const int* desca, double* af, const int* laf, double* work, const int* lwork, int* info);
void pdpbtrs(const char* uplo, const int* n, const int* bw, const int* nrhs, double* a, const int* ja, const int* desca, double* b, const int* ib, const int* descb, double* af, const int* laf, double* work, const int* lwork, int* info);
void pdposv(const char* uplo, const int* n, const int* nrhs, double* a, const int* ia, const int* ja, const int* desca, double* b, const int* ib, const int* jb, const int* descb, int* info);
void pdpotrf(const char* uplo, const int* n, double* a, const int* ia, const int* ja, const int* desca, int* info);
void pdpotrs(const char* uplo, const int* n, const int* nrhs, const double* a, const int* ia, const int* ja, const int* desca, double* b, const int* ib, const int* jb, const int* descb, int* info);
void psdbtrf(const int* n, const int* bwl, const int* bwu, float* a, const int* ja, const int* desca, float* af, const int* laf, float* work, const int* lwork, int* info);
void psdbtrs(const char* trans, const int* n, const int* bwl, const int* bwu, const int* nrhs, float* a, const int* ja, const int* desca, float* b, const int* ib, const int* descb, float* af, const int* laf, float* work, const int* lwork, int* info);
void psgbsv(const int* n, const int* bwl, const int* bwu, const int* nrhs, float* a, const int* ja, const int* desca, int* ipiv, float* b, const int* ib, const int* descb, float* work, const int* lwork, int* info);
void psgbtrf(const int* n, const int* bwl, const int* bwu, float* a, const int* ja, const int* desca, int* ipiv, float* af, const int* laf, float* work, const int* lwork, int* info);
void psgbtrs(const char* trans, const int* n, const int* bwl, const int* bwu, const int* nrhs, float* a, const int* ja, const int* desca, int* ipiv, float* b, const int* ib, const int* descb, float* af, const int* laf, float* work, const int* lwork, int* info);
void psgemr2d(const int* m, const int* n, const float* a, const int* ia, const int* ja, const int* desca, float* b, const int* ib, const int* jb, const int* descb, const int* ictxt);
void psgesv(const int* n, const int* nrhs, float* a, const int* ia, const int* ja, const int* desca, int* ipiv, float* b, const int* ib, const int* jb, const int* descb, int* info);
void psgetrf(const int* m, const int* n, float* a, const int* ia, const int* ja, const int* desca, int* ipiv, int* info);
void psgetrs(const char* trans, const int* n, const int* nrhs, const float* a, const int* ia, const int* ja, const int* desca, const int* ipiv, float* b, const int* ib, const int* jb, const int* descb, int* info);
void pspbtrf(const char* uplo, const int* n, const int* bw, float* a, const int* ja, const int* desca, float* af, const int* laf, float* work, const int* lwork, int* info);
void pspbtrs(const char* uplo, const int* n, const int* bw, const int* nrhs, float* a, const int* ja, const int* desca, float* b, const int* ib, const int* descb, float* af, const int* laf, float* work, const int* lwork, int* info);
void psposv(const char* uplo, const int* n, const int* nrhs, float* a, const int* ia, const int* ja, const int* desca, float* b, const int* ib, const int* jb, const int* descb, int* info);
void pspotrf(const char* uplo, const int* n, float* a, const int* ia, const int* ja, const int* desca, int* info);
void pspotrs(const char* uplo, const int* n, const int* nrhs, const float* a, const int* ia, const int* ja, const int* desca, float* b, const int* ib, const int* jb, const int* descb, int* info);
void pzgbtrf(const int* n, const int* bwl, const int* bwu, complex16* a, const int* ja, const int* desca, int* ipiv, complex16* af, const int* laf, complex16* work, const int* lwork, int* info);
void pzgbtrs(const char* trans, const int* n, const int* bwl, const int* bwu, const int* nrhs, complex16* a, const int* ja, const int* desca, int* ipiv, complex16* b, const int* ib, const int* descb, complex16* af, const int* laf, complex16* work, const int* lwork, int* info);
void pzgemr2d(const int* m, const int* n, const complex16* a, const int* ia, const int* ja, const int* desca, complex16* b, const int* ib, const int* jb, const int* descb, const int* ictxt);

void igamn2d(const int* ConTxt, const char* scope, const char* top, const int* m, const int* n, int* A, const int* lda, int* rA, int* cA, const int* ldia, const int* rdest, const int* cdest);
void igamx2d(const int* ConTxt, const char* scope, const char* top, const int* m, const int* n, int* A, const int* lda, int* rA, int* cA, const int* ldia, const int* rdest, const int* cdest);
void igebr2d(const int* ConTxt, const char* scope, const char* top, const int* m, const int* n, int* A, const int* lda, const int* rsrc, const int* csrc);

#ifdef __cplusplus
}
#endif

#endif // EZP_H