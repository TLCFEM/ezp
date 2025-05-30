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
      RECURSIVE SUBROUTINE CMUMPS_PROCESS_BLFAC_SLAVE(
     &   COMM_LOAD, ASS_IRECV,
     &   BUFR, LBUFR,
     &   LBUFR_BYTES, PROCNODE_STEPS, MSGSOU,
     &   SLAVEF, IWPOS, IWPOSCB, IPTRLU, LRLU, LRLUS, N, IW, LIW,
     &   A, LA, PTRIST, PTRAST, NSTK_S, PERM,
     &   COMP, STEP, PIMASTER, PAMASTER, POSFAC,
     &   MYID, COMM, IFLAG, IERROR, NBFIN,
     &
     &    PTLUST_S, PTRFAC, root, roota, OPASSW, OPELIW,
     &    ITLOC, RHS_MUMPS, FILS, DAD,
     &    PTRARW, PTRAIW,
     &    PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &    INTARR, DBLARR,
     &    ICNTL,KEEP,KEEP8,DKEEP,IPOOL, LPOOL, LEAF, ND, FRERE_STEPS,
     &    LPTRAR, NELT, FRTPTR, FRTELT, 
     &    ISTEP_TO_INIV2, TAB_POS_IN_PERE
     &               , LRGROUPS
     &    )
      USE MUMPS_BUF_COMMON, ONLY : MUMPS_BUF_SEND_1INT
      USE MUMPS_LOAD
      USE CMUMPS_LR_CORE
      USE CMUMPS_LR_TYPE
      USE CMUMPS_FAC_LR
      USE CMUMPS_LR_DATA_M
      USE MUMPS_INTR_TYPES, ONLY : MUMPS_ROOT_STRUC
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_ROOT_STRUC
      USE CMUMPS_DYNAMIC_MEMORY_M, ONLY : CMUMPS_DM_SET_DYNPTR
      USE CMUMPS_FAC_FRONT_AUX_M, ONLY : CMUMPS_GET_SIZE_SCHUR_IN_FRONT
#if ! defined(BLR_NOOPENMP)
!$    USE OMP_LIB
#endif
      IMPLICIT NONE
      TYPE (MUMPS_ROOT_STRUC) :: root
      TYPE (CMUMPS_ROOT_STRUC) :: roota
      INTEGER ICNTL( 60 ), KEEP( 500 )
      INTEGER(8) KEEP8(150)
      REAL       DKEEP(230)
      INTEGER LBUFR, LBUFR_BYTES
      INTEGER COMM_LOAD, ASS_IRECV
      INTEGER BUFR( LBUFR )
      INTEGER N, SLAVEF, IWPOS, IWPOSCB, LIW
      INTEGER(8) :: POSFAC, IPTRLU, LRLU, LRLUS, LA
      INTEGER(8) :: PTRAST(KEEP(28))
      INTEGER(8) :: PAMASTER(KEEP(28))
      INTEGER(8) :: PTRFAC(KEEP(28))
      INTEGER COMP
      INTEGER IFLAG, IERROR, NBFIN, MSGSOU
      INTEGER PROCNODE_STEPS(KEEP(28)), PTRIST(KEEP(28)),
     &        NSTK_S(KEEP(28))
      INTEGER PERM(N), STEP(N), PIMASTER(KEEP(28))
      INTEGER IW( LIW )
      COMPLEX A( LA )
      INTEGER, intent(in) :: LRGROUPS(KEEP(280))
      INTEGER NELT, LPTRAR
      INTEGER FRTPTR( N + 1 ), FRTELT( NELT )
      INTEGER(8), INTENT(IN) :: PTRAIW( LPTRAR ), PTRARW( LPTRAR )
      INTEGER(8), INTENT(IN) :: PTR8ARR(KEEP(193))
      INTEGER, INTENT(IN) :: NINCOLARR(KEEP(194))
      INTEGER, INTENT(IN) :: NINROWARR(KEEP(195))
      INTEGER, INTENT(IN) :: PTRDEBARR(KEEP(196))
      INTEGER ISTEP_TO_INIV2(KEEP(71)), 
     &        TAB_POS_IN_PERE(SLAVEF+2,max(1,KEEP(56)))
      INTEGER COMM, MYID
      INTEGER PTLUST_S(KEEP(28))
      INTEGER ITLOC( N + KEEP(253)), FILS( N ), DAD( KEEP(28) )
      COMPLEX :: RHS_MUMPS(KEEP8(85))
      INTEGER ND( KEEP(28) ), FRERE_STEPS( KEEP(28) )
      DOUBLE PRECISION OPASSW, OPELIW
      DOUBLE PRECISION FLOP1
      COMPLEX DBLARR( KEEP8(26) )
      INTEGER INTARR( KEEP8(27) )
      INTEGER LEAF, LPOOL 
      INTEGER IPOOL( LPOOL )
      INCLUDE 'mumps_headers.h'
      INCLUDE 'mpif.h'
      INCLUDE 'mumps_tags.h'
      INTEGER :: STATUS(MPI_STATUS_SIZE)
      INTEGER MUMPS_PROCNODE
      EXTERNAL MUMPS_PROCNODE
      INTEGER INODE, IPOSK, JPOSK, NCOLU, NPIV, POSITION, IERR
      INTEGER(8) POSELT, POSBLOCFACTO
      INTEGER(8) LAELL
      INTEGER(8) :: LA_PTR 
      COMPLEX, DIMENSION(:), POINTER :: A_PTR
      INTEGER IOLDPS, LCONT1, NROW1, NCOL1, NPIV1
      INTEGER NSLAV1 , HS, DEST, NSLAVES_FOLLOW
      INTEGER FPERE, TO_UPDATE_CPT_RECUR
      INTEGER(8) CPOS, LPOS
      LOGICAL DYNAMIC_ALLOC, COUNTER_WAS_HUGE
      LOGICAL BLOCKING, SET_IRECV, MESSAGE_RECEIVED
      LOGICAL LASTBL_INPANEL
      INTEGER allocok
      INTEGER LR_ACTIVATED_INT
      LOGICAL LR_ACTIVATED, COMPRESS_CB
      INTEGER NB_BLR_U, CURRENT_BLR_U
      TYPE (LRB_TYPE), DIMENSION(:), ALLOCATABLE :: BLR_U
      INTEGER, POINTER, DIMENSION(:) :: BEGS_BLR_U
      TYPE (LRB_TYPE), DIMENSION(:), POINTER :: BLR_LS
      TYPE (LRB_TYPE), POINTER, DIMENSION(:,:) :: CB_LRB
      INTEGER, POINTER, DIMENSION(:) :: BEGS_BLR_LS, BEGS_BLR_COL
      INTEGER    :: NB_BLR_LS, IPANEL,
     &           MAXI_CLUSTER_LS, MAXI_CLUSTER, 
     &           NB_BLR_COL, MAXI_CLUSTER_COL, NPARTSASS_MASTER
      COMPLEX, ALLOCATABLE, DIMENSION(:) :: WORK, TAU
      INTEGER, ALLOCATABLE, DIMENSION(:) :: JPVT
      COMPLEX, ALLOCATABLE, DIMENSION(:,:) :: BLOCKLR
      REAL,ALLOCATABLE,DIMENSION(:) :: RWORK
      INTEGER :: OMP_NUM, LWORK
      INTEGER :: II,JJ
      INTEGER ::  MSGSOU_BL
      INTEGER :: NFS4FATHER, NASS1, NELIM, INFO_TMP(2)
      INTEGER :: NVSCHUR_K253, NSLAVES_L, IROW_L
      INTEGER :: NBROWSinF
      REAL, ALLOCATABLE, DIMENSION(:) :: M_ARRAY
      COMPLEX, ALLOCATABLE, DIMENSION(:) :: UDYNAMIC
      COMPLEX ONE,ALPHA
      PARAMETER (ONE=(1.0E0,0.0E0), ALPHA=(-1.0E0,0.0E0))
      DYNAMIC_ALLOC = .FALSE.
      POSITION  = 0
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION, INODE, 1,
     &                 MPI_INTEGER, COMM, IERR )
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION, IPOSK, 1,
     &                 MPI_INTEGER, COMM, IERR )
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION, JPOSK, 1,
     &                 MPI_INTEGER, COMM, IERR )
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION, NPIV, 1,
     &                 MPI_INTEGER, COMM, IERR )
      IF ( NPIV .LE. 0 ) THEN
        NPIV = - NPIV
        WRITE(*,*) MYID,':error, received <=0 NPIV in BLFAC', NPIV
        CALL MUMPS_ABORT()
      END IF
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION, FPERE, 1,
     &                 MPI_INTEGER, COMM, IERR )
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION, NCOLU, 1,
     &                 MPI_INTEGER, COMM, IERR )
      LASTBL_INPANEL = (NCOLU.LT.0)
      IF (LASTBL_INPANEL) NCOLU = -NCOLU
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION, 
     &                 LR_ACTIVATED_INT, 1,
     &                 MPI_INTEGER, COMM, IERR )
      LR_ACTIVATED   = (LR_ACTIVATED_INT.EQ.1)
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION, 
     &                 IPANEL, 1,
     &                 MPI_INTEGER, COMM, IERR )
      IF (LR_ACTIVATED) THEN
          CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION,
     &                 NB_BLR_U, 1, MPI_INTEGER,
     &                 COMM, IERR )
          CURRENT_BLR_U = 1 
          ALLOCATE(BLR_U(max(NB_BLR_U,1)),
     &             BEGS_BLR_U(NB_BLR_U+2), stat=allocok)
          if (allocok .GT. 0) THEN
             IFLAG = -13 
             IERROR = max(NB_BLR_U,1) + NB_BLR_U+2
             GOTO 700
          endif
          CALL CMUMPS_MPI_UNPACK_LR(BUFR, LBUFR, LBUFR_BYTES,
     &                        POSITION, JPOSK-1, 0, 'V',
     &                        BLR_U, NB_BLR_U, 
     &                        BEGS_BLR_U(1),
     &                        KEEP8, COMM, IERR, IFLAG, IERROR)
          IF (IFLAG.LT.0) GOTO 700
      ELSE
      LAELL = int(NPIV,8) * int(NCOLU,8)
      CALL CMUMPS_GET_SIZE_NEEDED(
     &        0, LAELL, .FALSE.,
     &        KEEP(1), KEEP8(1),
     &        N, IW, LIW, A, LA,
     &        LRLU, IPTRLU,
     &        IWPOS, IWPOSCB, PTRIST, PTRAST,
     &        STEP, PIMASTER, PAMASTER, LRLUS,
     &        KEEP(IXSZ),COMP,DKEEP(97),MYID, SLAVEF,
     &        PROCNODE_STEPS, DAD, 
     &        IFLAG, IERROR)
      IF (IFLAG.LT.0) GOTO 700
      LRLU  = LRLU - LAELL
      LRLUS = LRLUS - LAELL
      KEEP8(67) = min(LRLUS, KEEP8(67))
      KEEP8(69) = KEEP8(69) + LAELL
      KEEP8(68) = max(KEEP8(69), KEEP8(68))
      POSBLOCFACTO = POSFAC
      POSFAC = POSFAC + LAELL
      CALL MUMPS_LOAD_MEM_UPDATE(.FALSE.,.FALSE.,
     &                           LA-LRLUS,0_8, LAELL,KEEP,KEEP8,LRLUS)
      CALL MPI_UNPACK( BUFR, LBUFR_BYTES, POSITION,
     &                 A(POSBLOCFACTO), NPIV*NCOLU,
     &                 MPI_COMPLEX,
     &                 COMM, IERR )
      ENDIF
      IF (PTRIST(STEP( INODE )) .EQ. 0) DYNAMIC_ALLOC = .TRUE.
      IF ( (PTRIST(STEP( INODE )).NE.0) .AND.
     &  (IPOSK + NPIV -1 .GT.
     &   IW(PTRIST(STEP(INODE))+3+KEEP(IXSZ))) )THEN
        DYNAMIC_ALLOC = .TRUE.
      ENDIF
      IF (LR_ACTIVATED) THEN
       DYNAMIC_ALLOC = .FALSE.
      ENDIF
      IF (DYNAMIC_ALLOC)  THEN
        ALLOCATE(UDYNAMIC(LAELL), stat=allocok)
        if (allocok .GT. 0) THEN
          IFLAG = -13 
          CALL MUMPS_SET_IERROR(LAELL,IERROR)
          GOTO 700
        endif
        UDYNAMIC(1_8:LAELL) = A(POSBLOCFACTO:POSBLOCFACTO+LAELL-1_8)
        LRLU  = LRLU + LAELL
        LRLUS = LRLUS + LAELL
        KEEP8(69) = KEEP8(69) - LAELL
        POSFAC = POSFAC - LAELL
      CALL MUMPS_LOAD_MEM_UPDATE(.FALSE.,.FALSE.,
     &          LA-LRLUS,0_8,-LAELL,KEEP,KEEP8,LRLUS)
      ENDIF
      IF (PTRIST(STEP( INODE )) .EQ. 0) THEN
          CALL CMUMPS_TREAT_DESCBAND( INODE, COMM_LOAD, ASS_IRECV,
     &      BUFR, LBUFR, LBUFR_BYTES, PROCNODE_STEPS, POSFAC,
     &      IWPOS, IWPOSCB, IPTRLU,
     &      LRLU, LRLUS, N, IW, LIW, A, LA, PTRIST,
     &      PTLUST_S, PTRFAC,
     &      PTRAST, STEP, PIMASTER, PAMASTER, NSTK_S, COMP,
     &      IFLAG, IERROR, COMM,
     &      PERM, IPOOL, LPOOL, LEAF, NBFIN, MYID, SLAVEF,
     &
     &      root, roota, OPASSW, OPELIW, ITLOC, RHS_MUMPS,
     &      FILS, DAD, PTRARW, PTRAIW,
     &      PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &      INTARR, DBLARR, ICNTL,KEEP,KEEP8,DKEEP,ND, FRERE_STEPS,
     &      LPTRAR, NELT, FRTPTR, FRTELT, 
     &      ISTEP_TO_INIV2, TAB_POS_IN_PERE, .TRUE.
     &               , LRGROUPS
     &     )
        IF ( IFLAG .LT. 0 ) GOTO 600
      ENDIF
      DO WHILE ( IPOSK + NPIV -1 .GT.
     &            IW( PTRIST(STEP( INODE )) + 3 +KEEP(IXSZ)) )
        MSGSOU_BL = MUMPS_PROCNODE( PROCNODE_STEPS(STEP(INODE)),
     &                           KEEP(199) )
        IOLDPS = PTRIST(STEP(INODE))
        NSLAVES_FOLLOW = IW( IOLDPS+5+KEEP(IXSZ))-XTRA_SLAVES_SYM
        NASS1 = abs(IW( IOLDPS + 1 + KEEP(IXSZ)))
        TO_UPDATE_CPT_RECUR =
     &                      ( SLAVEF - NSLAVES_FOLLOW - 1 ) *
     &                       (2*NASS1/KEEP(6))
        IF ( IW(IOLDPS+6+KEEP(IXSZ)) .eq.
     &        huge(IW(IOLDPS+6+KEEP(IXSZ))) ) THEN
              COUNTER_WAS_HUGE=.TRUE.
              IW(IOLDPS+6+KEEP(IXSZ)) = 1
        ELSE
              COUNTER_WAS_HUGE=.FALSE.
        ENDIF
        IW(IOLDPS+6+KEEP(IXSZ)) =
     &       IW(IOLDPS+6+KEEP(IXSZ)) - TO_UPDATE_CPT_RECUR - 10
        BLOCKING  = .TRUE.
        SET_IRECV = .FALSE.
        MESSAGE_RECEIVED = .FALSE.
        CALL CMUMPS_TRY_RECVTREAT( COMM_LOAD,
     &    ASS_IRECV, BLOCKING, SET_IRECV, MESSAGE_RECEIVED,
     &    MSGSOU_BL, BLOC_FACTO_SYM, STATUS, 
     &    BUFR, LBUFR, LBUFR_BYTES, PROCNODE_STEPS, POSFAC,
     &    IWPOS, IWPOSCB, IPTRLU,
     &    LRLU, LRLUS, N, IW, LIW, A, LA, PTRIST,
     &    PTLUST_S, PTRFAC,
     &    PTRAST, STEP, PIMASTER, PAMASTER, NSTK_S, COMP,
     &    IFLAG, IERROR, COMM,
     &    PERM, IPOOL, LPOOL, LEAF, NBFIN, MYID, SLAVEF,
     &
     &    root, roota, OPASSW, OPELIW, ITLOC, RHS_MUMPS,
     &    FILS, DAD, PTRARW, PTRAIW,
     &    PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &    INTARR, DBLARR, ICNTL, KEEP,KEEP8,DKEEP, ND, FRERE_STEPS,
     &    LPTRAR, NELT, FRTPTR, FRTELT,
     &    ISTEP_TO_INIV2, TAB_POS_IN_PERE, .TRUE.
     &               , LRGROUPS
     &      )
        IOLDPS = PTRIST(STEP(INODE))
        IW(IOLDPS+6+KEEP(IXSZ)) =
     &       IW(IOLDPS+6+KEEP(IXSZ)) + TO_UPDATE_CPT_RECUR + 10
        IF ( COUNTER_WAS_HUGE .AND.
     &       IW(IOLDPS+6+KEEP(IXSZ)).EQ.1 ) THEN
          IW(IOLDPS+6+KEEP(IXSZ)) = huge(IW(IOLDPS+6+KEEP(IXSZ)))
        ENDIF
        IF ( IFLAG .LT. 0 ) GOTO 600
      END DO
        SET_IRECV = .TRUE.
        BLOCKING  = .FALSE.
        MESSAGE_RECEIVED = .TRUE.
        CALL CMUMPS_TRY_RECVTREAT( COMM_LOAD,
     &    ASS_IRECV, BLOCKING, SET_IRECV, MESSAGE_RECEIVED,
     &    MPI_ANY_SOURCE, MPI_ANY_TAG, 
     &    STATUS, 
     &    BUFR, LBUFR, LBUFR_BYTES, PROCNODE_STEPS, POSFAC,
     &    IWPOS, IWPOSCB, IPTRLU,
     &    LRLU, LRLUS, N, IW, LIW, A, LA, PTRIST,
     &    PTLUST_S, PTRFAC,
     &    PTRAST, STEP, PIMASTER, PAMASTER, NSTK_S, COMP,
     &    IFLAG, IERROR, COMM,
     &    PERM, IPOOL, LPOOL, LEAF, NBFIN, MYID, SLAVEF,
     &
     &    root, roota, OPASSW, OPELIW, ITLOC, RHS_MUMPS,
     &    FILS, DAD, PTRARW, PTRAIW,
     &    PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &    INTARR, DBLARR, ICNTL, KEEP,KEEP8,DKEEP, ND, FRERE_STEPS,
     &    LPTRAR, NELT, FRTPTR, FRTELT, 
     &    ISTEP_TO_INIV2, TAB_POS_IN_PERE, .TRUE. 
     &               , LRGROUPS
     &     )
      IOLDPS  = PTRIST(STEP( INODE ))
      CALL CMUMPS_DM_SET_DYNPTR( IW(IOLDPS+XXS), A, LA,
     &     PTRAST(STEP(INODE)), IW(IOLDPS+XXD), IW(IOLDPS+XXR),
     &     A_PTR, POSELT, LA_PTR )
      LCONT1  = IW( IOLDPS + KEEP(IXSZ) )
      NROW1   = IW( IOLDPS + 2  + KEEP(IXSZ))
      NPIV1   = IW( IOLDPS + 3  + KEEP(IXSZ))
      NSLAV1  = IW( IOLDPS + 5  + KEEP(IXSZ))
      HS      = 6 + NSLAV1  + KEEP(IXSZ)
      NCOL1   = LCONT1 + NPIV1
      IF (LR_ACTIVATED) THEN
         CALL CMUMPS_BLR_DEC_AND_RETRIEVE_L (IW(IOLDPS+XXF), IPANEL, 
     &        BEGS_BLR_LS, BLR_LS, NCOLU)
         NB_BLR_LS = size(BEGS_BLR_LS)-2   
#if ! defined(BLR_NOOPENMP)
!$OMP PARALLEL
#endif
         CALL CMUMPS_BLR_UPDATE_TRAILING_I (
     &        A_PTR(POSELT), LA_PTR, 1_8,
     &        IFLAG, IERROR, NCOL1,
     &        BEGS_BLR_LS(1), size(BEGS_BLR_LS),
     &        BEGS_BLR_U(1), size(BEGS_BLR_U),
     &        CURRENT_BLR_U, 
     &        BLR_LS(1),  NB_BLR_LS+1,
     &        BLR_U(1), NB_BLR_U+1, 
     &        0,       
     &        .TRUE.,  
     &        0,       
     &        2,       
     &        1,
     &        KEEP(481), DKEEP(11), KEEP(466), KEEP(477) 
     &         )
#if ! defined(BLR_NOOPENMP)
!$OMP END PARALLEL
#endif          
          CALL DEALLOC_BLR_PANEL(BLR_U, NB_BLR_U, KEEP8, KEEP(34))
          IF (allocated(BLR_U)) DEALLOCATE(BLR_U)
          IF (associated(BEGS_BLR_U)) DEALLOCATE(BEGS_BLR_U)
          IF (IFLAG.LT.0) GOTO 700
         IF (KEEP(486).EQ.3) THEN
         CALL CMUMPS_BLR_TRY_FREE_PANEL(IW(IOLDPS+XXF), IPANEL, 
     &                            KEEP8, KEEP(34)) 
         ENDIF
      ELSE
      CPOS = POSELT + int(JPOSK - 1,8)
      LPOS = POSELT + int(IPOSK - 1,8)
      IF ( NPIV .GT. 0 ) THEN
          IF (DYNAMIC_ALLOC) THEN
            CALL cgemm( 'T', 'N', NCOLU, NROW1, NPIV, ALPHA,
     &            UDYNAMIC(1), NPIV,
     &            A_PTR( LPOS ), NCOL1, ONE,
     &            A_PTR( CPOS ), NCOL1 )
          ELSE
            CALL cgemm( 'T', 'N', NCOLU, NROW1, NPIV, ALPHA,
     &            A( POSBLOCFACTO ), NPIV,
     &            A_PTR( LPOS ), NCOL1, ONE,
     &            A_PTR( CPOS ), NCOL1 )
          ENDIF
      ENDIF 
      ENDIF
      IF (NPIV .GT. 0) THEN
        FLOP1 = dble(NCOLU*NPIV)*dble(2*NROW1)
        FLOP1 = -FLOP1
        CALL MUMPS_LOAD_UPDATE(1, .FALSE., FLOP1, KEEP,KEEP8 )
      ENDIF
      IF ( IW(IOLDPS+6+KEEP(IXSZ)).EQ.
     &    huge(IW(IOLDPS+6+KEEP(IXSZ))) ) THEN
          IW(IOLDPS+6+KEEP(IXSZ)) = 1
      ENDIF
      IF (LASTBL_INPANEL) THEN
         IW(IOLDPS+6+KEEP(IXSZ)) =
     &         IW(IOLDPS+6+KEEP(IXSZ)) + 1
      ENDIF
      IF (.NOT.LR_ACTIVATED) THEN
      IF (DYNAMIC_ALLOC) THEN
       DEALLOCATE(UDYNAMIC)
      ELSE
        LRLU  = LRLU + LAELL
        LRLUS = LRLUS + LAELL
        KEEP8(69) = KEEP8(69) - LAELL
        POSFAC = POSFAC - LAELL
      CALL MUMPS_LOAD_MEM_UPDATE(.FALSE.,.FALSE.,
     &                      LA-LRLUS,0_8,-LAELL,KEEP,KEEP8,LRLUS)
      ENDIF
      ENDIF
      NSLAVES_FOLLOW = IW( IOLDPS + 5 +KEEP(IXSZ) ) - XTRA_SLAVES_SYM
      IF ( IW( IOLDPS + 6  +KEEP(IXSZ)) .eq. 0 .and.
     &     KEEP(50) .ne. 0 .and. NSLAVES_FOLLOW .eq. 0 )
     &     THEN
         DEST = MUMPS_PROCNODE( PROCNODE_STEPS(STEP(INODE)), KEEP(199) )
         CALL MUMPS_BUF_SEND_1INT( INODE, DEST, END_NIV2_LDLT,
     &                             COMM, KEEP, IERR )
         IF ( IERR .LT. 0 ) THEN
           write(*,*) ' Internal error in PROCESS_BLFAC_SLAVE.'
           IFLAG = -99
           GOTO 700
         END IF
      END IF
      IF (IW(PTRIST(STEP(INODE)) + 6+KEEP(IXSZ) ) .eq. 0) THEN
           NPIV1 = IW( IOLDPS + 3  + KEEP(IXSZ))
           NASS1 = IW( IOLDPS + 4 + KEEP(IXSZ))  
           NELIM = NASS1 - NPIV1
          COMPRESS_CB= .FALSE.
          IF (LR_ACTIVATED) THEN
            COMPRESS_CB = ((IW(PTRIST(STEP(INODE))+XXLR).EQ.1).OR.
     &                     (IW(PTRIST(STEP(INODE))+XXLR).EQ.3))
            IF (NPIV.EQ.0) CALL MUMPS_ABORT()
            IF (COMPRESS_CB) THEN
              CALL CMUMPS_BLR_RETRIEVE_BEGS_BLR_C (IW(IOLDPS+XXF), 
     &                  BEGS_BLR_COL,  NPARTSASS_MASTER)
              NB_BLR_COL   = size(BEGS_BLR_COL) - 1
              allocate(CB_LRB(NB_BLR_LS,NB_BLR_COL-NPARTSASS_MASTER),
     &                 stat=allocok)
              IF (allocok > 0) THEN
                IFLAG  = -13
                IERROR = NB_BLR_LS*(NB_BLR_COL-NPARTSASS_MASTER)
                GOTO 700
              ENDIF
              DO II=1,NB_BLR_LS
              DO JJ=1,NB_BLR_COL-NPARTSASS_MASTER
                CB_LRB(II,JJ)%M=0
                CB_LRB(II,JJ)%N=0
                NULLIFY(CB_LRB(II,JJ)%Q)
                NULLIFY(CB_LRB(II,JJ)%R)
                CB_LRB(II,JJ)%ISLR = .FALSE.
              ENDDO
              ENDDO
              CALL CMUMPS_BLR_SAVE_CB_LRB(IW(IOLDPS+XXF),CB_LRB)
              call MAX_CLUSTER(BEGS_BLR_LS,NB_BLR_LS+1,MAXI_CLUSTER_LS)
              CALL MAX_CLUSTER(BEGS_BLR_COL( max(NPARTSASS_MASTER,1)+1:
     &        NB_BLR_COL+1),
     &        NB_BLR_COL-max(NPARTSASS_MASTER,1), MAXI_CLUSTER_COL )
              MAXI_CLUSTER = max(MAXI_CLUSTER_LS,
     &         MAXI_CLUSTER_COL+NELIM,NPIV)
              LWORK = MAXI_CLUSTER*(MAXI_CLUSTER+1)
              OMP_NUM = 1
#if ! defined(BLR_NOOPENMP)
!$            OMP_NUM = OMP_GET_MAX_THREADS()
#endif
              ALLOCATE(BLOCKLR(MAXI_CLUSTER, OMP_NUM*MAXI_CLUSTER),
     &            RWORK(2*MAXI_CLUSTER*OMP_NUM), 
     &            TAU(MAXI_CLUSTER*OMP_NUM),
     &            JPVT(MAXI_CLUSTER*OMP_NUM), 
     &            WORK(LWORK*OMP_NUM),
     &            stat=allocok)
              IF (allocok > 0 ) THEN
                IFLAG  = -13
                IERROR = OMP_NUM*(LWORK + MAXI_CLUSTER*(MAXI_CLUSTER+4))
                GOTO 700
              ENDIF
              NFS4FATHER = -9999
              IF ( (KEEP(219).NE.0).AND.(KEEP(50).EQ.2) ) THEN
               CALL CMUMPS_BLR_RETRIEVE_NFS4FATHER ( IW(IOLDPS+XXF),
     &             NFS4FATHER )
               NFS4FATHER = max(NFS4FATHER,0) + NELIM
              ENDIF
              ALLOCATE(M_ARRAY(max(NFS4FATHER,1)), stat=allocok)
              IF (allocok.gt.0) THEN
                IFLAG = -13
                IERROR = max(NFS4FATHER,1)
                GOTO 700
              ENDIF
              BEGS_BLR_COL(1+NPARTSASS_MASTER) = 
     &               BEGS_BLR_COL(1+NPARTSASS_MASTER) - NELIM
              NBROWSinF = 0
              IF ( (KEEP(219).NE.0).AND.(KEEP(50).EQ.2).AND.
     &             NFS4FATHER.GT.0  ) THEN
                CALL CMUMPS_COMPUTE_NBROWSinF (
     &                N, INODE, FPERE, KEEP, 
     &                IOLDPS, HS, 
     &                IW, LIW, 
     &                NROW1, NCOL1, NPIV1,
     &                NELIM, NFS4FATHER,
     &                NBROWSinF
     &                )
              ENDIF
              IF ((KEEP(114).EQ.1) .AND. (KEEP(116).GT.0)
     &            .AND. (KEEP(50).EQ.2)
     &           ) THEN
                 NSLAVES_L = IW(PTRIST(STEP(INODE)) + 5 + KEEP(IXSZ))
                 IROW_L    = PTRIST(STEP(INODE)) + 6 + NSLAVES_L + 
     &                       KEEP(IXSZ)
                 CALL CMUMPS_GET_SIZE_SCHUR_IN_FRONT ( 
     &                N, 
     &                NROW1,
     &                KEEP(116), 
     &                IW(IROW_L),
     &                PERM, NVSCHUR_K253 )
              ELSE
                 NVSCHUR_K253 = 0
              ENDIF
#if ! defined(BLR_NOOPENMP)
!$OMP PARALLEL
#endif
              CALL CMUMPS_COMPRESS_CB_I(
     &          A_PTR(POSELT), LA_PTR, 1_8, NCOL1,
     &          BEGS_BLR_LS(1), size(BEGS_BLR_LS),
     &          BEGS_BLR_COL(1), size(BEGS_BLR_COL),
     &          NB_BLR_LS, NB_BLR_COL-NPARTSASS_MASTER,
     &          NPARTSASS_MASTER, 
     &          NROW1, NCOL1-NPIV1, INODE,
     &          IW(IOLDPS+XXF), 1, 2, IFLAG, IERROR,
     &          DKEEP(12), KEEP(466), KEEP(484), KEEP(489),
     &          CB_LRB(1,1),
     &          WORK, TAU, JPVT, LWORK, RWORK, BLOCKLR,
     &          MAXI_CLUSTER, KEEP8, OMP_NUM,
     &          NFS4FATHER, NPIV1, NVSCHUR_K253, KEEP(1), 
     &          M_ARRAY,
     &          NELIM, NBROWSinF )
#if ! defined(BLR_NOOPENMP)
!$OMP END PARALLEL
#endif
              IF (IFLAG.LT.0) GOTO 650
              IF ( (KEEP(219).NE.0).AND.(KEEP(50).EQ.2).AND.
     &             NFS4FATHER.GT.0  ) THEN
                 INFO_TMP(1) = IFLAG
                 INFO_TMP(2) = IERROR
                 CALL CMUMPS_BLR_SAVE_M_ARRAY( IW(IOLDPS+XXF),
     &            M_ARRAY, INFO_TMP)
                 IFLAG  = INFO_TMP(1) 
                 IERROR = INFO_TMP(2) 
              ENDIF
 650          CONTINUE         
              IF (allocated(M_ARRAY)) DEALLOCATE(M_ARRAY)
              IF (allocated(BLOCKLR)) DEALLOCATE(BLOCKLR)
              IF (allocated(RWORK)) DEALLOCATE(RWORK)
              IF (allocated(TAU)) DEALLOCATE(TAU)
              IF (allocated(JPVT)) DEALLOCATE(JPVT)
              IF (allocated(WORK)) DEALLOCATE(WORK)           
              IF (IFLAG.LT.0) GOTO 700
            ENDIF 
          ENDIF
         CALL CMUMPS_END_FACTO_SLAVE( COMM_LOAD, ASS_IRECV, 
     &    N, INODE, FPERE, 
     &    root, roota,
     &    MYID, COMM,
     &    
     &    BUFR, LBUFR, LBUFR_BYTES, PROCNODE_STEPS, POSFAC,
     &    IWPOS, IWPOSCB, IPTRLU, LRLU, LRLUS, IW, LIW, A, LA,
     &    PTRIST, PTLUST_S, PTRFAC,
     &    PTRAST, STEP, PIMASTER, PAMASTER,
     &    NSTK_S, COMP, IFLAG, IERROR, PERM,
     &    IPOOL, LPOOL, LEAF, NBFIN, SLAVEF,
     &    OPASSW, OPELIW, ITLOC, RHS_MUMPS, FILS, DAD, PTRARW, PTRAIW,
     &    PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &    INTARR,DBLARR,ICNTL,KEEP,KEEP8,DKEEP,ND,FRERE_STEPS,
     &    LPTRAR, NELT, FRTPTR, FRTELT, 
     &    ISTEP_TO_INIV2, TAB_POS_IN_PERE 
     &               , LRGROUPS
     &     )
       ENDIF 
      RETURN
 700  CONTINUE
      CALL CMUMPS_BDC_ERROR( MYID, SLAVEF, COMM, KEEP )
 600  CONTINUE
      IF (allocated(BLR_U)) DEALLOCATE(BLR_U)
      IF (COMPRESS_CB) THEN
        IF (allocated(BLOCKLR)) DEALLOCATE(BLOCKLR)
        IF (allocated(RWORK)) DEALLOCATE(RWORK)
        IF (allocated(TAU)) DEALLOCATE(TAU)
        IF (allocated(JPVT)) DEALLOCATE(JPVT)
        IF (allocated(WORK)) DEALLOCATE(WORK)
      ENDIF
      IF (allocated(M_ARRAY)) DEALLOCATE(M_ARRAY)
      IF (DYNAMIC_ALLOC) THEN
       IF (allocated(UDYNAMIC)) DEALLOCATE(UDYNAMIC)
      ENDIF
      RETURN
      END SUBROUTINE CMUMPS_PROCESS_BLFAC_SLAVE
