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
#include <cstdint>

#ifdef EZP_INT64
using int_t = std::int64_t;
#else
using int_t = std::int32_t;
#endif

using complex8 = std::complex<float>;
using complex16 = std::complex<double>;

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

#define pcdbtrf EZP(pcdbtrf)
#define pcdbtrs EZP(pcdbtrs)
#define pcgbtrf EZP(pcgbtrf)
#define pcgbtrs EZP(pcgbtrs)
#define pcgemr2d EZP(pcgemr2d)
#define pcgesvx EZP(pcgesvx)
#define pcgetrf EZP(pcgetrf)
#define pcgetrs EZP(pcgetrs)
#define pcpbtrf EZP(pcpbtrf)
#define pcpbtrs EZP(pcpbtrs)
#define pcposvx EZP(pcposvx)
#define pcpotrf EZP(pcpotrf)
#define pcpotrs EZP(pcpotrs)
#define pddbtrf EZP(pddbtrf)
#define pddbtrs EZP(pddbtrs)
#define pdgbsv EZP(pdgbsv)
#define pdgbtrf EZP(pdgbtrf)
#define pdgbtrs EZP(pdgbtrs)
#define pdgemr2d EZP(pdgemr2d)
#define pdgesv EZP(pdgesv)
#define pdgesvx EZP(pdgesvx)
#define pdgetrf EZP(pdgetrf)
#define pdgetrs EZP(pdgetrs)
#define pdpbtrf EZP(pdpbtrf)
#define pdpbtrs EZP(pdpbtrs)
#define pdposv EZP(pdposv)
#define pdposvx EZP(pdposvx)
#define pdpotrf EZP(pdpotrf)
#define pdpotrs EZP(pdpotrs)
#define psdbtrf EZP(psdbtrf)
#define psdbtrs EZP(psdbtrs)
#define psgbsv EZP(psgbsv)
#define psgbtrf EZP(psgbtrf)
#define psgbtrs EZP(psgbtrs)
#define psgemr2d EZP(psgemr2d)
#define psgesv EZP(psgesv)
#define psgesvx EZP(psgesvx)
#define psgetrf EZP(psgetrf)
#define psgetrs EZP(psgetrs)
#define pspbtrf EZP(pspbtrf)
#define pspbtrs EZP(pspbtrs)
#define psposv EZP(psposv)
#define psposvx EZP(psposvx)
#define pspotrf EZP(pspotrf)
#define pspotrs EZP(pspotrs)
#define pzdbtrf EZP(pzdbtrf)
#define pzdbtrs EZP(pzdbtrs)
#define pzgbtrf EZP(pzgbtrf)
#define pzgbtrs EZP(pzgbtrs)
#define pzgemr2d EZP(pzgemr2d)
#define pzgesvx EZP(pzgesvx)
#define pzgetrf EZP(pzgetrf)
#define pzgetrs EZP(pzgetrs)
#define pzpbtrf EZP(pzpbtrf)
#define pzpbtrs EZP(pzpbtrs)
#define pzposvx EZP(pzposvx)
#define pzpotrf EZP(pzpotrf)
#define pzpotrs EZP(pzpotrs)

#define igamn2d EZP(igamn2d)
#define igamx2d EZP(igamx2d)
#define igebr2d EZP(igebr2d)
#define pigemr2d EZP(pigemr2d)

#ifdef __cplusplus
extern "C" {
#endif

int_t numroc(const int_t* n, const int_t* nb, const int_t* iproc, const int_t* isrcproc, const int_t* nprocs);

void blacs_exit(const int_t* notDone);
void blacs_get(const int_t* ConTxt, const int_t* what, int_t* val);
void blacs_gridexit(const int_t* ConTxt);
void blacs_gridinfo(const int_t* ConTxt, int_t* nprow, int_t* npcol, int_t* myrow, int_t* mycol);
void blacs_gridinit(int_t* ConTxt, const char* layout, const int_t* nprow, const int_t* npcol);
void blacs_pinfo(int_t* mypnum, int_t* nprocs);

void descinit(int_t* desc, const int_t* m, const int_t* n, const int_t* mb, const int_t* nb, const int_t* irsrc, const int_t* icsrc, const int_t* ictxt, const int_t* lld, int_t* info);

void pcdbtrf(const int_t* n, const int_t* bwl, const int_t* bwu, complex8* a, const int_t* ja, const int_t* desca, complex8* af, const int_t* laf, complex8* work, const int_t* lwork, int_t* info);
void pcdbtrs(const char* trans, const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, complex8* a, const int_t* ja, const int_t* desca, complex8* b, const int_t* ib, const int_t* descb, complex8* af, const int_t* laf, complex8* work, const int_t* lwork, int_t* info);
void pcgbtrf(const int_t* n, const int_t* bwl, const int_t* bwu, complex8* a, const int_t* ja, const int_t* desca, int_t* ipiv, complex8* af, const int_t* laf, complex8* work, const int_t* lwork, int_t* info);
void pcgbtrs(const char* trans, const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, complex8* a, const int_t* ja, const int_t* desca, int_t* ipiv, complex8* b, const int_t* ib, const int_t* descb, complex8* af, const int_t* laf, complex8* work, const int_t* lwork, int_t* info);
void pcgemr2d(const int_t* m, const int_t* n, const complex8* a, const int_t* ia, const int_t* ja, const int_t* desca, complex8* b, const int_t* ib, const int_t* jb, const int_t* descb, const int_t* ictxt);
void pcgesvx(const char* fact, const char* trans, const int_t* n, const int_t* nrhs, complex8* a, const int_t* ia, const int_t* ja, const int_t* desca, complex8* af, const int_t* iaf, const int_t* jaf, const int_t* descaf, int_t* ipiv, char* equed, float* r, float* c, complex8* b, const int_t* ib, const int_t* jb, const int_t* descb, complex8* x, const int_t* ix, const int_t* jx, const int_t* descx, float* rcond, float* ferr, float* berr, complex8* work, const int_t* lwork, float* rwork, const int_t* lrwork, int_t* info);
void pcgetrf(const int_t* m, const int_t* n, complex8* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* ipiv, int_t* info);
void pcgetrs(const char* trans, const int_t* n, const int_t* nrhs, const complex8* a, const int_t* ia, const int_t* ja, const int_t* desca, const int_t* ipiv, complex8* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void pcpbtrf(const char* uplo, const int_t* n, const int_t* bw, complex8* a, const int_t* ja, const int_t* desca, complex8* af, const int_t* laf, complex8* work, const int_t* lwork, int_t* info);
void pcpbtrs(const char* uplo, const int_t* n, const int_t* bw, const int_t* nrhs, complex8* a, const int_t* ja, const int_t* desca, complex8* b, const int_t* ib, const int_t* descb, complex8* af, const int_t* laf, complex8* work, const int_t* lwork, int_t* info);
void pcposvx(const char* fact, const char* uplo, const int_t* n, const int_t* nrhs, complex8* a, const int_t* ia, const int_t* ja, const int_t* desca, complex8* af, const int_t* iaf, const int_t* jaf, const int_t* descaf, char* equed, float* sr, float* sc, complex8* b, const int_t* ib, const int_t* jb, const int_t* descb, complex8* x, const int_t* ix, const int_t* jx, const int_t* descx, float* rcond, float* ferr, float* berr, complex8* work, const int_t* lwork, float* rwork, const int_t* lrwork, int_t* info);
void pcpotrf(const char* uplo, const int_t* n, complex8* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* info);
void pcpotrs(const char* uplo, const int_t* n, const int_t* nrhs, const complex8* a, const int_t* ia, const int_t* ja, const int_t* desca, complex8* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void pddbtrf(const int_t* n, const int_t* bwl, const int_t* bwu, double* a, const int_t* ja, const int_t* desca, double* af, const int_t* laf, double* work, const int_t* lwork, int_t* info);
void pddbtrs(const char* trans, const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, double* a, const int_t* ja, const int_t* desca, double* b, const int_t* ib, const int_t* descb, double* af, const int_t* laf, double* work, const int_t* lwork, int_t* info);
void pdgbsv(const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, double* a, const int_t* ja, const int_t* desca, int_t* ipiv, double* b, const int_t* ib, const int_t* descb, double* work, const int_t* lwork, int_t* info);
void pdgbtrf(const int_t* n, const int_t* bwl, const int_t* bwu, double* a, const int_t* ja, const int_t* desca, int_t* ipiv, double* af, const int_t* laf, double* work, const int_t* lwork, int_t* info);
void pdgbtrs(const char* trans, const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, double* a, const int_t* ja, const int_t* desca, int_t* ipiv, double* b, const int_t* ib, const int_t* descb, double* af, const int_t* laf, double* work, const int_t* lwork, int_t* info);
void pdgemr2d(const int_t* m, const int_t* n, const double* a, const int_t* ia, const int_t* ja, const int_t* desca, double* b, const int_t* ib, const int_t* jb, const int_t* descb, const int_t* ictxt);
void pdgesv(const int_t* n, const int_t* nrhs, double* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* ipiv, double* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void pdgesvx(const char* fact, const char* trans, const int_t* n, const int_t* nrhs, double* a, const int_t* ia, const int_t* ja, const int_t* desca, double* af, const int_t* iaf, const int_t* jaf, const int_t* descaf, int_t* ipiv, char* equed, double* r, double* c, double* b, const int_t* ib, const int_t* jb, const int_t* descb, double* x, const int_t* ix, const int_t* jx, const int_t* descx, double* rcond, double* ferr, double* berr, double* work, const int_t* lwork, int_t* iwork, const int_t* liwork, int_t* info);
void pdgetrf(const int_t* m, const int_t* n, double* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* ipiv, int_t* info);
void pdgetrs(const char* trans, const int_t* n, const int_t* nrhs, const double* a, const int_t* ia, const int_t* ja, const int_t* desca, const int_t* ipiv, double* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void pdpbtrf(const char* uplo, const int_t* n, const int_t* bw, double* a, const int_t* ja, const int_t* desca, double* af, const int_t* laf, double* work, const int_t* lwork, int_t* info);
void pdpbtrs(const char* uplo, const int_t* n, const int_t* bw, const int_t* nrhs, double* a, const int_t* ja, const int_t* desca, double* b, const int_t* ib, const int_t* descb, double* af, const int_t* laf, double* work, const int_t* lwork, int_t* info);
void pdposv(const char* uplo, const int_t* n, const int_t* nrhs, double* a, const int_t* ia, const int_t* ja, const int_t* desca, double* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void pdposvx(const char* fact, const char* uplo, const int_t* n, const int_t* nrhs, double* a, const int_t* ia, const int_t* ja, const int_t* desca, double* af, const int_t* iaf, const int_t* jaf, const int_t* descaf, char* equed, double* sr, double* sc, double* b, const int_t* ib, const int_t* jb, const int_t* descb, double* x, const int_t* ix, const int_t* jx, const int_t* descx, double* rcond, double* ferr, double* berr, double* work, const int_t* lwork, int_t* iwork, const int_t* liwork, int_t* info);
void pdpotrf(const char* uplo, const int_t* n, double* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* info);
void pdpotrs(const char* uplo, const int_t* n, const int_t* nrhs, const double* a, const int_t* ia, const int_t* ja, const int_t* desca, double* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void psdbtrf(const int_t* n, const int_t* bwl, const int_t* bwu, float* a, const int_t* ja, const int_t* desca, float* af, const int_t* laf, float* work, const int_t* lwork, int_t* info);
void psdbtrs(const char* trans, const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, float* a, const int_t* ja, const int_t* desca, float* b, const int_t* ib, const int_t* descb, float* af, const int_t* laf, float* work, const int_t* lwork, int_t* info);
void psgbsv(const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, float* a, const int_t* ja, const int_t* desca, int_t* ipiv, float* b, const int_t* ib, const int_t* descb, float* work, const int_t* lwork, int_t* info);
void psgbtrf(const int_t* n, const int_t* bwl, const int_t* bwu, float* a, const int_t* ja, const int_t* desca, int_t* ipiv, float* af, const int_t* laf, float* work, const int_t* lwork, int_t* info);
void psgbtrs(const char* trans, const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, float* a, const int_t* ja, const int_t* desca, int_t* ipiv, float* b, const int_t* ib, const int_t* descb, float* af, const int_t* laf, float* work, const int_t* lwork, int_t* info);
void psgemr2d(const int_t* m, const int_t* n, const float* a, const int_t* ia, const int_t* ja, const int_t* desca, float* b, const int_t* ib, const int_t* jb, const int_t* descb, const int_t* ictxt);
void psgesv(const int_t* n, const int_t* nrhs, float* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* ipiv, float* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void psgesvx(const char* fact, const char* trans, const int_t* n, const int_t* nrhs, float* a, const int_t* ia, const int_t* ja, const int_t* desca, float* af, const int_t* iaf, const int_t* jaf, const int_t* descaf, int_t* ipiv, char* equed, float* r, float* c, float* b, const int_t* ib, const int_t* jb, const int_t* descb, float* x, const int_t* ix, const int_t* jx, const int_t* descx, float* rcond, float* ferr, float* berr, float* work, const int_t* lwork, int_t* iwork, const int_t* liwork, int_t* info);
void psgetrf(const int_t* m, const int_t* n, float* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* ipiv, int_t* info);
void psgetrs(const char* trans, const int_t* n, const int_t* nrhs, const float* a, const int_t* ia, const int_t* ja, const int_t* desca, const int_t* ipiv, float* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void pspbtrf(const char* uplo, const int_t* n, const int_t* bw, float* a, const int_t* ja, const int_t* desca, float* af, const int_t* laf, float* work, const int_t* lwork, int_t* info);
void pspbtrs(const char* uplo, const int_t* n, const int_t* bw, const int_t* nrhs, float* a, const int_t* ja, const int_t* desca, float* b, const int_t* ib, const int_t* descb, float* af, const int_t* laf, float* work, const int_t* lwork, int_t* info);
void psposv(const char* uplo, const int_t* n, const int_t* nrhs, float* a, const int_t* ia, const int_t* ja, const int_t* desca, float* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void psposvx(const char* fact, const char* uplo, const int_t* n, const int_t* nrhs, float* a, const int_t* ia, const int_t* ja, const int_t* desca, float* af, const int_t* iaf, const int_t* jaf, const int_t* descaf, char* equed, float* sr, float* sc, float* b, const int_t* ib, const int_t* jb, const int_t* descb, float* x, const int_t* ix, const int_t* jx, const int_t* descx, float* rcond, float* ferr, float* berr, float* work, const int_t* lwork, int_t* iwork, const int_t* liwork, int_t* info);
void pspotrf(const char* uplo, const int_t* n, float* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* info);
void pspotrs(const char* uplo, const int_t* n, const int_t* nrhs, const float* a, const int_t* ia, const int_t* ja, const int_t* desca, float* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void pzdbtrf(const int_t* n, const int_t* bwl, const int_t* bwu, complex16* a, const int_t* ja, const int_t* desca, complex16* af, const int_t* laf, complex16* work, const int_t* lwork, int_t* info);
void pzdbtrs(const char* trans, const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, complex16* a, const int_t* ja, const int_t* desca, complex16* b, const int_t* ib, const int_t* descb, complex16* af, const int_t* laf, complex16* work, const int_t* lwork, int_t* info);
void pzgbtrf(const int_t* n, const int_t* bwl, const int_t* bwu, complex16* a, const int_t* ja, const int_t* desca, int_t* ipiv, complex16* af, const int_t* laf, complex16* work, const int_t* lwork, int_t* info);
void pzgbtrs(const char* trans, const int_t* n, const int_t* bwl, const int_t* bwu, const int_t* nrhs, complex16* a, const int_t* ja, const int_t* desca, int_t* ipiv, complex16* b, const int_t* ib, const int_t* descb, complex16* af, const int_t* laf, complex16* work, const int_t* lwork, int_t* info);
void pzgemr2d(const int_t* m, const int_t* n, const complex16* a, const int_t* ia, const int_t* ja, const int_t* desca, complex16* b, const int_t* ib, const int_t* jb, const int_t* descb, const int_t* ictxt);
void pzgesvx(const char* fact, const char* trans, const int_t* n, const int_t* nrhs, complex16* a, const int_t* ia, const int_t* ja, const int_t* desca, complex16* af, const int_t* iaf, const int_t* jaf, const int_t* descaf, int_t* ipiv, char* equed, double* r, double* c, complex16* b, const int_t* ib, const int_t* jb, const int_t* descb, complex16* x, const int_t* ix, const int_t* jx, const int_t* descx, double* rcond, double* ferr, double* berr, complex16* work, const int_t* lwork, double* rwork, const int_t* lrwork, int_t* info);
void pzgetrf(const int_t* m, const int_t* n, complex16* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* ipiv, int_t* info);
void pzgetrs(const char* trans, const int_t* n, const int_t* nrhs, const complex16* a, const int_t* ia, const int_t* ja, const int_t* desca, const int_t* ipiv, complex16* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);
void pzpbtrf(const char* uplo, const int_t* n, const int_t* bw, complex16* a, const int_t* ja, const int_t* desca, complex16* af, const int_t* laf, complex16* work, const int_t* lwork, int_t* info);
void pzpbtrs(const char* uplo, const int_t* n, const int_t* bw, const int_t* nrhs, complex16* a, const int_t* ja, const int_t* desca, complex16* b, const int_t* ib, const int_t* descb, complex16* af, const int_t* laf, complex16* work, const int_t* lwork, int_t* info);
void pzposvx(const char* fact, const char* uplo, const int_t* n, const int_t* nrhs, complex16* a, const int_t* ia, const int_t* ja, const int_t* desca, complex16* af, const int_t* iaf, const int_t* jaf, const int_t* descaf, char* equed, double* sr, double* sc, complex16* b, const int_t* ib, const int_t* jb, const int_t* descb, complex16* x, const int_t* ix, const int_t* jx, const int_t* descx, double* rcond, double* ferr, double* berr, complex16* work, const int_t* lwork, double* rwork, const int_t* lrwork, int_t* info);
void pzpotrf(const char* uplo, const int_t* n, complex16* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* info);
void pzpotrs(const char* uplo, const int_t* n, const int_t* nrhs, const complex16* a, const int_t* ia, const int_t* ja, const int_t* desca, complex16* b, const int_t* ib, const int_t* jb, const int_t* descb, int_t* info);

void igamn2d(const int_t* ConTxt, const char* scope, const char* top, const int_t* m, const int_t* n, int_t* A, const int_t* lda, int_t* rA, int_t* cA, const int_t* ldia, const int_t* rdest, const int_t* cdest);
void igamx2d(const int_t* ConTxt, const char* scope, const char* top, const int_t* m, const int_t* n, int_t* A, const int_t* lda, int_t* rA, int_t* cA, const int_t* ldia, const int_t* rdest, const int_t* cdest);
void igebr2d(const int_t* ConTxt, const char* scope, const char* top, const int_t* m, const int_t* n, int_t* A, const int_t* lda, const int_t* rsrc, const int_t* csrc);
void pigemr2d(const int_t* m, const int_t* n, const int_t* a, const int_t* ia, const int_t* ja, const int_t* desca, int_t* b, const int_t* ib, const int_t* jb, const int_t* descb, const int_t* ictxt);

void cluster_sparse_solver(void*, const std::int32_t*, const std::int32_t*, const std::int32_t*, const std::int32_t*, const std::int32_t*, const void*, const std::int32_t*, const std::int32_t*, std::int32_t*, const std::int32_t*, std::int32_t*, const std::int32_t*, void*, void*, const std::int32_t*, std::int32_t*);
void cluster_sparse_solver_64(void*, const std::int64_t*, const std::int64_t*, const std::int64_t*, const std::int64_t*, const std::int64_t*, const void*, const std::int64_t*, const std::int64_t*, std::int64_t*, const std::int64_t*, std::int64_t*, const std::int64_t*, void*, void*, const std::int32_t*, std::int64_t*);

#ifdef __cplusplus
}
#endif

#endif // EZP_H
