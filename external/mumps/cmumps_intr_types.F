C
C  This file is part of MUMPS 5.8.0, released
C  on Tue May  6 08:27:40 UTC 2025
C
C
C  Copyright 1991-2025 CERFACS, CNRS, ENS Lyon, INP Toulouse, Inria,
C  Mumps Technologies, University of Bordeaux.
C
C  This version of MUMPS is provided to you free of charge. It is
C  released under the CeCILL-C license 
C  (see doc/CeCILL-C_V1-en.txt, doc/CeCILL-C_V1-fr.txt, and
C  https://cecill.info/licences/Licence_CeCILL-C_V1-en.html)
C
      MODULE CMUMPS_INTR_TYPES
      USE MUMPS_INTR_TYPES, ONLY: MUMPS_ROOT_STRUC
      PRIVATE
      PUBLIC :: CMUMPS_ROOT_STRUC,
     &          CMUMPS_L0OMPFAC_T,
     &          CMUMPS_INTR_STRUC,
     &          CMUMPS_INIT_INTR_ENCODING,
     &          CMUMPS_FREE_INTR_ENCODING,
     &          CMUMPS_ENCODE_INTR,
     &          CMUMPS_DECODE_INTR
C     CMUMPS_ROOT_STRUC no longer contains INTEGERS
      TYPE CMUMPS_ROOT_STRUC
!       Centralized master of root
        COMPLEX, DIMENSION(:), POINTER :: RHS_CNTR_MASTER_ROOT
!       Used to access Schur easily from root structure
        COMPLEX, DIMENSION(:), POINTER :: SCHUR_POINTER
!       for try_null_space preprocessing constant only:
        COMPLEX, DIMENSION(:), POINTER :: QR_TAU
!       Fwd in facto: 
!           case of scalapack root: to store RHS in 2D block cyclic
!           format compatible with root distribution
        COMPLEX, DIMENSION(:,:), POINTER :: RHS_ROOT
!       for SVD on root (#define try_null_space)
        COMPLEX, DIMENSION(:,:), POINTER :: SVD_U, SVD_VT
!       for RR on root (#define try_null_space)
        REAL, DIMENSION(:), POINTER :: SINGULAR_VALUES
!
      END TYPE CMUMPS_ROOT_STRUC
!     multicore
      TYPE CMUMPS_L0OMPFAC_T
         COMPLEX, POINTER, DIMENSION(:) :: A
         INTEGER(8) :: LA
      END TYPE CMUMPS_L0OMPFAC_T
C
C     All MUMPS internal datatypes are in an internal structure:
      TYPE CMUMPS_INTR_STRUC
         TYPE (MUMPS_ROOT_STRUC)  :: root
         TYPE (CMUMPS_ROOT_STRUC) :: roota
         TYPE (CMUMPS_L0OMPFAC_T), DIMENSION(:), POINTER ::
     &                             L0_OMP_FACTORS
      END TYPE CMUMPS_INTR_STRUC
C     =================================================================
      CONTAINS
C     =================================================================
      SUBROUTINE CMUMPS_INIT_INTR_ENCODING(id_intr_ENCODING)
      IMPLICIT NONE
      CHARACTER, DIMENSION(:), POINTER :: id_intr_ENCODING
C     To be called only before JOB=-1
      NULLIFY(id_intr_ENCODING)
      END SUBROUTINE CMUMPS_INIT_INTR_ENCODING
C     =================================================================
      SUBROUTINE CMUMPS_FREE_INTR_ENCODING(id_intr_ENCODING)
      IMPLICIT NONE
      CHARACTER, DIMENSION(:), POINTER :: id_intr_ENCODING
C     To be called only after JOB=-2
      DEALLOCATE(id_intr_ENCODING)
      NULLIFY(id_intr_ENCODING)
      RETURN
      END SUBROUTINE CMUMPS_FREE_INTR_ENCODING
C     =================================================================
      SUBROUTINE CMUMPS_ENCODE_INTR(id_intr_ENCODING, id_intr)
      IMPLICIT NONE
C
C     Arguments:
C     =========
      CHARACTER, DIMENSION(:), POINTER :: id_intr_ENCODING
      TYPE (CMUMPS_INTR_STRUC)         :: id_intr
C
C     Local variables:
C     ===============
      CHARACTER :: CHAR_ARRAY(1)
      INTEGER :: CHAR_LENGTH, IERR
C
      IF (associated(id_intr_ENCODING)) THEN
C       Should be unassociated on entry
        WRITE(*,*) "Internal error in CMUMPS_ENCODE_INTR:",
     &             " id_intr_ENCODING already allocated"
        CALL MUMPS_ABORT()
      ENDIF
      CHAR_LENGTH=size(transfer(id_intr,CHAR_ARRAY))
      ALLOCATE(id_intr_ENCODING(CHAR_LENGTH), stat=IERR)
      IF (IERR > 0 ) THEN
        WRITE(*,*) "Allocation error in MUMPS_ENCODE_INTR"
        CALL MUMPS_ABORT()
      ENDIF
C     Fill with derived datatype
      id_intr_ENCODING=transfer(id_intr,CHAR_ARRAY)
      RETURN
      END SUBROUTINE CMUMPS_ENCODE_INTR
C     =================================================================
      SUBROUTINE CMUMPS_DECODE_INTR(id_intr_ENCODING, id_intr)
      IMPLICIT NONE
      CHARACTER(len=1), DIMENSION(:), POINTER :: id_intr_ENCODING
      TYPE (CMUMPS_INTR_STRUC) :: id_intr
      IF (.NOT.associated(id_intr_ENCODING)) THEN
        WRITE(*,*) "Internal error 1 in CMUMPS_DECODE_INTR"
        CALL MUMPS_ABORT()
      ENDIf
      id_intr=transfer(id_intr_ENCODING,id_intr)
      DEALLOCATE(id_intr_ENCODING)
      NULLIFY(id_intr_ENCODING)
      RETURN
      END SUBROUTINE CMUMPS_DECODE_INTR
      END MODULE CMUMPS_INTR_TYPES
