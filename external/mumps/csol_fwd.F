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
      SUBROUTINE CMUMPS_SOL_R(N, A, LA, IW, LIW, WCB, LWCB,
     &    NRHS,
     &    PTRICB, IWCB, LIWCB, 
     &    RHSINTR, LRHSINTR, POSINRHSINTR_FWD, 
     &    STEP,
     &    FRERE, DAD, FILS,
     &    NSTK, IPOOL, LPOOL, PTRIST, PTRFAC, MYLEAF, MYROOT,
     &    INFO,
     &    KEEP, KEEP8, DKEEP,
     &    PROCNODE_STEPS,
     &    SLAVEF, COMM, MYID, BUFR, LBUFR, LBUFR_BYTES,
     &    RHS_ROOT, LRHS_ROOT, MTYPE, 
     &
     &    ISTEP_TO_INIV2, TAB_POS_IN_PERE
     &    , RHS_BOUNDS, LRHS_BOUNDS, DO_NBSPARSE, FROM_PP
     &    , L0_OMP_MAPPING, LL0_OMP_MAPPING,
     &    L0_OMP_FACTORS, LL0_OMP_FACTORS
     &    )
      USE CMUMPS_STATIC_PTR_M, ONLY : CMUMPS_SET_STATIC_PTR,
     &                                CMUMPS_GET_TMP_PTR
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_L0OMPFAC_T
      IMPLICIT NONE
      INTEGER MTYPE
      INTEGER(8), INTENT(IN) :: LA, LWCB
      INTEGER, INTENT(IN) :: N, LIW, LPOOL, LIWCB
      INTEGER, INTENT(IN) :: SLAVEF, MYLEAF, MYROOT, COMM, MYID
      INTEGER INFO( 80 ), KEEP(500)
      INTEGER(8) KEEP8(150)
      REAL, INTENT(INOUT) :: DKEEP(230)
      INTEGER PROCNODE_STEPS( KEEP(28) )
      INTEGER NRHS
      COMPLEX A( LA ), WCB( LWCB )
      INTEGER(8), intent(in) :: LRHS_ROOT
      COMPLEX RHS_ROOT( LRHS_ROOT )
      INTEGER LBUFR, LBUFR_BYTES
      INTEGER BUFR( LBUFR )
      INTEGER STEP( N ), FRERE( KEEP(28) ), FILS( N ),
     &        DAD( KEEP(28) )
      INTEGER NSTK(KEEP(28)), IPOOL( LPOOL )
      INTEGER PTRIST(KEEP(28))
      INTEGER(8) :: PTRFAC(KEEP(28))
      INTEGER PTRICB( KEEP(28) ) 
      LOGICAL, intent(in) :: DO_NBSPARSE
      INTEGER, intent(in) :: LRHS_BOUNDS
      INTEGER, intent(in) :: RHS_BOUNDS(LRHS_BOUNDS)
      INTEGER IW( LIW ), IWCB( LIWCB )
      INTEGER ISTEP_TO_INIV2(KEEP(71)), 
     &        TAB_POS_IN_PERE(SLAVEF+2,max(1,KEEP(56)))
      INTEGER, intent(in) ::  POSINRHSINTR_FWD(N), LRHSINTR 
      COMPLEX, intent(inout) :: RHSINTR(LRHSINTR,NRHS)
      LOGICAL, intent(in) :: FROM_PP
      INTEGER, INTENT( in ) :: LL0_OMP_MAPPING, LL0_OMP_FACTORS
      INTEGER, INTENT( in ) :: L0_OMP_MAPPING( LL0_OMP_MAPPING )
      TYPE (CMUMPS_L0OMPFAC_T), INTENT(IN) ::
     &                        L0_OMP_FACTORS(LL0_OMP_FACTORS)
      INCLUDE 'mpif.h'
      INCLUDE 'mumps_tags.h'
      INTEGER DUMMY(1)
      LOGICAL FLAG
      COMPLEX, DIMENSION(:), POINTER :: A_PTR
      INTEGER(8) :: LA_PTR
      INTEGER :: UNDERL0MAP 
      INTEGER NBFIN, MYROOT_LEFT
      INTEGER POSIWCB
      INTEGER(8) :: POSWCB, PLEFTWCB
      INTEGER INODE, IFATH
      INTEGER III, LEAF
      LOGICAL BLOQ
      EXTERNAL MUMPS_PROCNODE
      INTEGER MUMPS_PROCNODE
      LOGICAL ERROR_WAS_BROADCASTED
      DUMMY(1) = 1
      KEEP(266)=0
      POSIWCB = LIWCB
      POSWCB  = LWCB
      PLEFTWCB= 1_8
      PTRICB = 0
      LEAF = MYLEAF + 1
      III    = 1
      NBFIN = SLAVEF
      MYROOT_LEFT = MYROOT
      IF ( MYROOT_LEFT .EQ. 0 ) THEN
        NBFIN = NBFIN - 1
        CALL CMUMPS_MCAST2(DUMMY, 1, MPI_INTEGER, MYID, COMM,
     &       RACINE_SOLVE, SLAVEF, KEEP)
        IF (NBFIN.EQ.0) GOTO 260
      END IF
      IF ( INFO(1) .LT. 0 ) THEN
        GOTO 260
      ENDIF
   50 CONTINUE
      IF (SLAVEF .EQ. 1) THEN
         CALL CMUMPS_GET_INODE_FROM_POOL
     &        ( IPOOL(1), LPOOL, III, LEAF, INODE,
     &          KEEP(208) )
        GOTO 60
      ENDIF
      IF (SLAVEF .EQ. 1) THEN
        FLAG = .FALSE.
      ELSE
        BLOQ = ( ( III .EQ. LEAF )
     &     )
        CALL CMUMPS_SOLVE_RECV_AND_TREAT( BLOQ, FLAG,
     &       BUFR, LBUFR, LBUFR_BYTES,
     &       MYID, SLAVEF, COMM,
     &       N, NRHS, IPOOL, LPOOL, LEAF,
     &       NBFIN, NSTK, IW, LIW, A, LA, PTRIST, PTRFAC,
     &       IWCB, LIWCB,
     &       WCB, LWCB, POSWCB,
     &       PLEFTWCB, POSIWCB,
     &       PTRICB, INFO, KEEP,KEEP8, DKEEP, STEP,
     &       PROCNODE_STEPS,
     &       RHSINTR, LRHSINTR, POSINRHSINTR_FWD
     &       , FROM_PP
     &      )
      ENDIF
      IF ( INFO( 1 ) .LT. 0 .OR. NBFIN .EQ. 0 ) GOTO 260
      IF (.not. FLAG) THEN
         IF (III .NE. LEAF) THEN
            CALL CMUMPS_GET_INODE_FROM_POOL
     &           (IPOOL(1), LPOOL, III, LEAF, INODE,
     &           KEEP(208) )
            GOTO 60
         ENDIF                  
      ENDIF                     
      GOTO 50
 60   CONTINUE
      IF (KEEP(400) .GT. 0 ) THEN
        UNDERL0MAP = L0_OMP_MAPPING(STEP(INODE))
      ELSE
        UNDERL0MAP = 0
      ENDIF
      IF (UNDERL0MAP .EQ. 0 .OR. KEEP(201).GT.0) THEN
        CALL CMUMPS_SET_STATIC_PTR(A)
        CALL CMUMPS_GET_TMP_PTR(A_PTR)
        LA_PTR = LA
      ELSE
        A_PTR => L0_OMP_FACTORS(UNDERL0MAP)%A
        LA_PTR = L0_OMP_FACTORS(UNDERL0MAP)%LA
      ENDIF
      CALL CMUMPS_SOLVE_NODE_FWD( INODE,
     &        huge(INODE), huge(INODE), 
     &        BUFR, LBUFR, LBUFR_BYTES,
     &        MYID, SLAVEF, COMM,  N,
     &        IPOOL, LPOOL, LEAF, NBFIN, NSTK,
     &        IWCB, LIWCB, WCB, LWCB, A_PTR(1), LA_PTR,
     &        IW, LIW, NRHS, 
     &        POSWCB, PLEFTWCB, POSIWCB,
     &        PTRICB, PTRIST, PTRFAC, PROCNODE_STEPS,
     &        FILS, STEP, FRERE, DAD,
     &        INFO, KEEP,KEEP8, DKEEP, RHS_ROOT, LRHS_ROOT, MTYPE,
     &        RHSINTR, LRHSINTR, POSINRHSINTR_FWD,
     &        ISTEP_TO_INIV2, TAB_POS_IN_PERE
     &        , RHS_BOUNDS, LRHS_BOUNDS, DO_NBSPARSE 
     &        , FROM_PP
     &        , ERROR_WAS_BROADCASTED
     & )
      IF ( INFO(1) .LT. 0 ) THEN
        IF (.NOT. ERROR_WAS_BROADCASTED) THEN
          CALL CMUMPS_BDC_ERROR( MYID, SLAVEF, COMM, KEEP )
        ENDIF
        GOTO 260
      ENDIF
      IFATH = DAD(STEP(INODE))
      IF ( IFATH .EQ. 0 ) THEN
        MYROOT_LEFT = MYROOT_LEFT - 1
        IF (MYROOT_LEFT .EQ. 0) THEN
          NBFIN = NBFIN - 1
          IF (SLAVEF .GT. 1) THEN
            CALL CMUMPS_MCAST2(DUMMY, 1, MPI_INTEGER, MYID,
     &           COMM, RACINE_SOLVE, SLAVEF, KEEP)
          ENDIF
        END IF
      ELSE
        IF ( MUMPS_PROCNODE(PROCNODE_STEPS(STEP(IFATH)), KEEP(199))
     &       .EQ. MYID ) THEN
           IF ( PTRICB(STEP(INODE)) .EQ. 1 .OR.
     &          PTRICB(STEP(INODE)) .EQ. -1 ) THEN
             NSTK(STEP(IFATH)) = NSTK(STEP(IFATH)) - 1
             IF (NSTK(STEP(IFATH)) .EQ. 0) THEN
               IPOOL(LEAF) = IFATH
               LEAF = LEAF + 1
               IF (LEAF .GT. LPOOL) THEN
                  WRITE(*,*)
     &            'Internal error CMUMPS_TRAITER_MESSAGE_SOLVE',
     &            LEAF, LPOOL
                  CALL MUMPS_ABORT()
               ENDIF
             ENDIF
             PTRICB(STEP(INODE)) = 0
           ENDIF
        ENDIF
      ENDIF
      IF ( NBFIN .EQ. 0 ) GOTO 260
      GOTO 50
  260 CONTINUE
      CALL MUMPS_CLEAN_PENDING(INFO(1), KEEP, BUFR, LBUFR,LBUFR_BYTES,
     &     COMM, DUMMY(1),  
     &     SLAVEF, .TRUE., .FALSE.) 
      RETURN
      END SUBROUTINE CMUMPS_SOL_R
