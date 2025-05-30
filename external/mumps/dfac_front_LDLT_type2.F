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
      MODULE DMUMPS_FAC2_LDLT_M
      CONTAINS
      SUBROUTINE DMUMPS_FAC2_LDLT( COMM_LOAD, ASS_IRECV, 
     &           N, INODE, FPERE, IW, LIW, A, LA,
     &           UU, NNEGW, NNULLNEGW, NPVW, NB22T2W, NBTINYW,
     &           DET_EXPW, DET_MANTW, DET_SIGNW,
     &             COMM, MYID, BUFR, LBUFR,LBUFR_BYTES,NBFIN,LEAF,
     &             IFLAG, IERROR, IPOOL,LPOOL,
     &             SLAVEF, POSFAC, IWPOS, IWPOSCB, IPTRLU, LRLU,
     &             LRLUS, COMP,
     &             PTRIST, PTRAST, PTLUST_S, PTRFAC, STEP,
     &             PIMASTER, PAMASTER,
     &             NSTK_S,PERM,PROCNODE_STEPS, root, roota,
     &             OPASSW, OPELIW, ITLOC, RHS_MUMPS,
     &             FILS, DAD, PTRARW, PTRAIW,
     &             PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &             INTARR, DBLARR, ICNTL, KEEP,KEEP8, ND, FRERE,
     &             LPTRAR, NELT, FRTPTR, FRTELT, SEUIL,
     &             ISTEP_TO_INIV2, TAB_POS_IN_PERE, AVOID_DELAYED,
     &             DKEEP, PIVNUL_LIST_STRUCT
     &     , LRGROUPS
     &             )
      USE DMUMPS_FAC_FRONT_AUX_M
      USE DMUMPS_FAC_FRONT_TYPE2_AUX_M
      USE DMUMPS_OOC
      USE DMUMPS_FAC_LR
      USE DMUMPS_LR_TYPE
      USE MUMPS_LR_STATS
      USE DMUMPS_ANA_LR, ONLY : GET_CUT
      USE DMUMPS_LR_DATA_M
!$    USE OMP_LIB
      USE MUMPS_INTR_TYPES, ONLY : MUMPS_ROOT_STRUC
      USE DMUMPS_INTR_TYPES, ONLY : DMUMPS_ROOT_STRUC
      USE MUMPS_BUF_COMMON, ONLY : MUMPS_BUF_TEST
      USE MUMPS_PIVNUL_MOD, ONLY : PIVNUL_LIST_STRUCT_T
      IMPLICIT NONE
      INTEGER COMM_LOAD, ASS_IRECV
      INTEGER N, INODE, FPERE, LIW
      INTEGER, intent(inout) :: NNEGW, NPVW, NNULLNEGW, NB22T2W, NBTINYW
      INTEGER, intent(inout) :: DET_EXPW, DET_SIGNW
      DOUBLE PRECISION, intent(inout) :: DET_MANTW
      INTEGER(8) :: LA
      INTEGER, TARGET :: IW( LIW )
      DOUBLE PRECISION A( LA )
      DOUBLE PRECISION UU, SEUIL
      TYPE (MUMPS_ROOT_STRUC) :: root
      TYPE (DMUMPS_ROOT_STRUC) :: roota
      INTEGER COMM, MYID, LBUFR, LBUFR_BYTES
      INTEGER LPTRAR, NELT
      INTEGER ICNTL(60), KEEP(500)
      INTEGER(8) KEEP8(150)
      INTEGER NBFIN, SLAVEF,
     &        IFLAG, IERROR, LEAF, LPOOL
      INTEGER(8) :: POSFAC, IPTRLU, LRLU, LRLUS
      INTEGER IWPOS, IWPOSCB, COMP 
      INTEGER NB_BLOC_FAC
      INTEGER FRTPTR( N + 1 ), FRTELT( NELT )
      INTEGER BUFR( LBUFR ), IPOOL(LPOOL),
     &        ITLOC(N+KEEP(253)), FILS(N), DAD( KEEP(28) ),
     &        ND( KEEP(28) ), FRERE( KEEP(28) )
      INTEGER(8), INTENT(IN) :: PTRARW(LPTRAR), PTRAIW(LPTRAR)
      INTEGER(8), INTENT(IN) :: PTR8ARR(KEEP(193))
      INTEGER, INTENT(IN) :: NINCOLARR(KEEP(194))
      INTEGER, INTENT(IN) :: NINROWARR(KEEP(195))
      INTEGER, INTENT(IN) :: PTRDEBARR(KEEP(196))
      DOUBLE PRECISION :: RHS_MUMPS(KEEP8(85))
      INTEGER(8) :: PTRAST(KEEP(28))
      INTEGER(8) :: PTRFAC(KEEP(28))
      INTEGER(8) :: PAMASTER(KEEP(28))
      INTEGER PTRIST(KEEP(28)), PTLUST_S(KEEP(28)),
     &        STEP(N), PIMASTER(KEEP(28)),
     &        NSTK_S(KEEP(28)), PERM(N),
     &        PROCNODE_STEPS(KEEP(28))
      INTEGER ISTEP_TO_INIV2(KEEP(71)), 
     &        TAB_POS_IN_PERE(SLAVEF+2,max(1,KEEP(56)))
      DOUBLE PRECISION OPASSW, OPELIW
      DOUBLE PRECISION DBLARR(KEEP8(26))
      INTEGER INTARR(KEEP8(27))
      LOGICAL AVOID_DELAYED
      TYPE(PIVNUL_LIST_STRUCT_T)     :: PIVNUL_LIST_STRUCT
      DOUBLE PRECISION DKEEP(230)
      INTEGER :: LRGROUPS(KEEP(280))
      INTEGER(8) :: POSELT
      INTEGER IOLDPS, allocok, K263,J
      INTEGER INOPV, IFINB, NFRONT, NPIV, IEND_BLOCK
      INTEGER NASS, LDAFS, IBEG_BLOCK 
      INTEGER :: NB_POSTPONED
      INTEGER :: IBEG_BLOCK_FOR_IPIV
      LOGICAL LASTPANEL, LR_ACTIVATED, COMPRESS_PANEL
      LOGICAL OOCWRITE_COMPATIBLE_WITH_BLR, 
     &        OOC_EFFECTIVE_ON_FRONT, 
     &        OOC_EFF_AND_WRITE_BYPANEL
      INTEGER NBLR_ORIG, IBEG_BLR, IEND_BLR, CURRENT_BLR
      INTEGER Inextpiv
      LOGICAL RESET_TO_ONE
      INTEGER K109_SAVE
      INTEGER XSIZE, NBKJIB_ORIG
      DOUBLE PRECISION UUTEMP
      INCLUDE 'mumps_headers.h'
      INTEGER , ALLOCATABLE, DIMENSION ( : ) :: IPIV
      DOUBLE PRECISION , ALLOCATABLE, DIMENSION ( : )    :: DIAG_ORIG
      INTEGER    :: SIZEDIAG_ORIG
      INTEGER(8) :: LAFAC
      INTEGER LIWFAC, STRAT, TYPEFile, NextPiv2beWritten,
     &        IDUMMY, NELIM
      TYPE(IO_BLOCK) :: MonBloc 
      LOGICAL LAST_CALL
      INTEGER PP_FIRST2SWAP_L, IFLAG_OOC
      INTEGER PP_LastPIVRPTRFilled 
      INTEGER INFO_TMP(2)
      INTEGER :: MAXI_RANK
      INTEGER HF, NPARTSASS, NPARTSCB, NB_BLR
      INTEGER MAXI_CLUSTER, LWORK
      TYPE(LRB_TYPE), DIMENSION(1), TARGET  :: BLR_DUMMY
      INTEGER, POINTER, DIMENSION(:)        :: PTDummy
      TYPE(LRB_TYPE), POINTER, DIMENSION(:) :: ACC_LUA
      INTEGER, POINTER, DIMENSION(:)        :: BEGS_BLR
      TYPE(LRB_TYPE), POINTER, DIMENSION(:) :: BLR_L, BLR_SEND
      DOUBLE PRECISION, POINTER, DIMENSION(:)        :: DIAG
      TYPE(LRB_TYPE), POINTER, DIMENSION(:) :: BLR_PANEL
      INTEGER, POINTER, DIMENSION(:) :: BEGS_BLR_TMP, BEGS_BLR_STATIC
      INTEGER :: DIAGSIZ_STA, DIAGSIZ_DYN, DPOS, I, IP, MEM, MEM_TOT
      INTEGER(8) :: POSELT_DIAG, APOSMAX
      DOUBLE PRECISION, ALLOCATABLE :: WORK(:), TAU(:)
      INTEGER, ALLOCATABLE :: JPVT(:)
      DOUBLE PRECISION,ALLOCATABLE :: RWORK(:)
      DOUBLE PRECISION, ALLOCATABLE :: BLOCK(:,:)
      INTEGER :: OMP_NUM
      LOGICAL :: SWAP_OCCURRED
      INTEGER :: MY_NUM
      INTEGER PIVOT_OPTION
      INTEGER LAST_ROW
      EXTERNAL DMUMPS_BDC_ERROR
      LOGICAL STATICMODE
      DOUBLE PRECISION SEUIL_LOC
      DOUBLE PRECISION GW_FACTCUMUL
      INTEGER PIVSIZ,IWPOSPIV
      DOUBLE PRECISION ONE
      PARAMETER (ONE = 1.0D0)
      NULLIFY(PTDummy)
      NULLIFY(ACC_LUA)
      NULLIFY(BEGS_BLR)
      NULLIFY(BLR_L) 
      NULLIFY(BLR_SEND)
      NULLIFY(DIAG)
      NULLIFY(BLR_PANEL)
      NULLIFY(BEGS_BLR_TMP)
      NULLIFY(BEGS_BLR_STATIC)
      IF (KEEP(206).GE.1) THEN
        Inextpiv = 1   
      ELSE 
        Inextpiv = 0   
      ENDIF
      SWAP_OCCURRED = .FALSE.
      INOPV = 0
      IF(KEEP(97) .EQ. 0) THEN
         STATICMODE = .FALSE.
      ELSE
         STATICMODE = .TRUE.
      ENDIF
      IF (AVOID_DELAYED) THEN
        STATICMODE = .TRUE.
        UUTEMP=UU
        SEUIL_LOC = max(SEUIL,epsilon(SEUIL))
      ELSE
        SEUIL_LOC=SEUIL
        UUTEMP=UU
      ENDIF
      RESET_TO_ONE = ((KEEP(110).GT.0).AND.(DKEEP(2).LE.0.0D0))
      IF (RESET_TO_ONE) THEN
        K109_SAVE = KEEP(109)
      ENDIF
      IBEG_BLOCK  = 1
      NB_BLOC_FAC = 0
      XSIZE  = KEEP(IXSZ)
      IOLDPS = PTLUST_S(STEP( INODE ))
      POSELT = PTRAST(STEP( INODE ))
      NFRONT = IW(IOLDPS+XSIZE)
      NASS   = abs(IW(IOLDPS+2+XSIZE))
      LDAFS  = NASS
      IF ((KEEP(219).EQ.1).AND.(KEEP(207).EQ.1).AND.(KEEP(50).EQ.2)
     &   ) THEN
        APOSMAX = POSELT + int(LDAFS,8)*int(LDAFS,8)
        NB_POSTPONED = max(NFRONT - ND(STEP(INODE)),0)
        CALL DMUMPS_UPDATE_PARPIV_ENTRIES ( INODE,
     &     KEEP, A(APOSMAX), NASS, NB_POSTPONED)
      ENDIF
      IW(IOLDPS+3+XSIZE) =  -99999
      LR_ACTIVATED= .FALSE. 
      LR_ACTIVATED   = (IW(IOLDPS+XXLR).GT.0)
      COMPRESS_PANEL = (IW(IOLDPS+XXLR).GE.2)
      OOCWRITE_COMPATIBLE_WITH_BLR = 
     &          ( .NOT.LR_ACTIVATED.OR.  (.NOT.COMPRESS_PANEL).OR.
     &            (KEEP(486).NE.2) 
     &          )
      OOC_EFFECTIVE_ON_FRONT= ((KEEP(201).EQ.1).AND. 
     &                         OOCWRITE_COMPATIBLE_WITH_BLR)
      CALL DMUMPS_SET_INNERBLOCKSIZE( NBKJIB_ORIG, NASS, KEEP)
      IF (.not.LR_ACTIVATED) THEN
          NBLR_ORIG     = KEEP(420)
      ELSE
          NBLR_ORIG  = -9999 
      ENDIF
      IF (LR_ACTIVATED) THEN
         K263 = 1   
      ELSE
         K263 = KEEP(263)
         IF (K263 .NE. 0 .AND. NASS/NBLR_ORIG < 4) THEN
           IF ( NBLR_ORIG .GT. NBKJIB_ORIG * 4 ) THEN
             NBLR_ORIG = max(NBKJIB_ORIG, (NASS+3)/4)
           ELSE
             K263 = 0
           ENDIF
         ENDIF
      ENDIF
      PIVOT_OPTION = MIN(2,KEEP(468))
      IF ((UUTEMP == 0.0D0) .AND. OOC_EFFECTIVE_ON_FRONT) THEN
          IF (K263.EQ.1.AND.(.NOT.LR_ACTIVATED)) THEN
            PIVOT_OPTION = 0
          ENDIF
      ENDIF
      IEND_BLOCK  = 0
      IEND_BLR    = 0
      CURRENT_BLR = 0
      ALLOCATE( IPIV( NASS ), stat = allocok )
      IF ( allocok .GT. 0 ) THEN
        WRITE(*,*) MYID, ' : DMUMPS_FAC2_LDLT failed to allocate ',
     &  NASS, ' integers'
        IFLAG = -13
        IERROR=NASS
        GO TO 500
      END IF
      IF (KEEP(219).GE.3) THEN
       SIZEDIAG_ORIG = NASS
      ELSE
       SIZEDIAG_ORIG = 1
      ENDIF
      ALLOCATE ( DIAG_ORIG(SIZEDIAG_ORIG), stat = allocok )
      IF ( allocok .GT. 0 ) THEN
          WRITE(*,*) MYID,
     &      ' : FAC_NIV2 failed to allocate ',
     &      NASS, ' REAL/COMPLEX entries'
          IFLAG=-13
          IERROR=NASS
          GO TO 500
      END IF
      CALL MUMPS_GETI8(LAFAC,IW(IOLDPS+XXR))
      LIWFAC    = IW(IOLDPS+XXI)
      IF (OOC_EFFECTIVE_ON_FRONT) THEN
        IDUMMY    = -9876
        TYPEFile  = TYPEF_L
        NextPiv2beWritten = 1 
        PP_FIRST2SWAP_L = NextPiv2beWritten 
        MonBloc%LastPanelWritten_L = 0 
        MonBloc%INODE    = INODE
        MonBloc%MASTER   = .TRUE.
        MonBloc%Typenode = 2
        MonBloc%NROW     = NASS
        MonBloc%NCOL     = NASS
        MonBloc%NFS      = NASS
        MonBloc%Last     = .FALSE.   
        MonBloc%LastPiv  = -66666    
        MonBloc%INDICES =>
     &  IW(IOLDPS+6+NFRONT+XSIZE+IW(IOLDPS+5+XSIZE)
     &    :IOLDPS+5+2*NFRONT+XSIZE+IW(IOLDPS+5+XSIZE))
      ENDIF
      IF (LR_ACTIVATED) THEN
             IF (KEEP(475).EQ.3) THEN
               IF (UUTEMP == 0.0D0) THEN
                 PIVOT_OPTION = 0
               ELSE
                 PIVOT_OPTION = 1
               ENDIF
             ENDIF
             CNT_NODES = CNT_NODES + 1 
      ENDIF
      HF = 6 + IW(IOLDPS+5+XSIZE)+XSIZE
      OOC_EFF_AND_WRITE_BYPANEL  = ( (PIVOT_OPTION.GE.2) .AND.
     &                                     OOC_EFFECTIVE_ON_FRONT )
      IF (LR_ACTIVATED) THEN
         CALL GET_CUT(IW(IOLDPS+HF:IOLDPS+HF+NFRONT-1), NASS,
     &        0, LRGROUPS, NPARTSCB, 
     &        NPARTSASS, BEGS_BLR)
         CALL REGROUPING2(BEGS_BLR, NPARTSASS, NASS, NPARTSCB,
     &        0, KEEP(488), .FALSE., KEEP(472), 
     &        NFRONT, KEEP(1))     
         NB_BLR = NPARTSASS + NPARTSCB
         call MAX_CLUSTER(BEGS_BLR,NB_BLR,MAXI_CLUSTER)
         MAXI_RANK = KEEP(479)*MAXI_CLUSTER
         LWORK = MAXI_CLUSTER*(MAXI_CLUSTER+1)
         OMP_NUM = 1
#if ! defined(BLR_NOOPENMP)
!$       OMP_NUM = OMP_GET_MAX_THREADS()
#endif
         ALLOCATE(BLOCK(MAXI_CLUSTER, OMP_NUM*MAXI_CLUSTER),
     &             RWORK(2*MAXI_CLUSTER*OMP_NUM), 
     &             TAU(MAXI_CLUSTER*OMP_NUM),
     &             JPVT(MAXI_CLUSTER*OMP_NUM), 
     &             WORK(LWORK*OMP_NUM),stat=allocok)
         IF (allocok > 0) THEN
           IFLAG  = -13
           IERROR = OMP_NUM*(LWORK + MAXI_CLUSTER*(MAXI_CLUSTER+4))
           GOTO 480
         ENDIF
         ALLOCATE(ACC_LUA(OMP_NUM),stat=allocok)
         IF (allocok > 0) THEN
           IFLAG  = -13
           IERROR = OMP_NUM
           GOTO 480
         ENDIF
         IF (KEEP(480).GE.3) THEN
           DO MY_NUM=1,OMP_NUM
             CALL ALLOC_LRB(ACC_LUA(MY_NUM), MAXI_RANK,
     &                      MAXI_CLUSTER, MAXI_CLUSTER, .TRUE.,
     &                      IFLAG, IERROR, KEEP8)
             IF (IFLAG.LT.0) GOTO 480
             ACC_LUA(MY_NUM)%K = 0
           ENDDO
         ENDIF
      ENDIF
      IF (LR_ACTIVATED.AND.(KEEP(480).NE.0
     &       .OR.
     &       (
     &         (KEEP(486).EQ.2) 
     &       )
     &      )) THEN
        INFO_TMP(1) = IFLAG
        INFO_TMP(2) = IERROR
        CALL DMUMPS_BLR_INIT_FRONT(IW(IOLDPS+XXF), INFO_TMP)
        IFLAG  = INFO_TMP(1) 
        IERROR = INFO_TMP(2) 
        IF (IFLAG.LT.0) GOTO 500
        CALL DMUMPS_BLR_SAVE_INIT(IW(IOLDPS+XXF), 
     &              .TRUE., 
     &              .TRUE., 
     &              .FALSE., 
     &              NPARTSASS, 
     &              BEGS_BLR, PTDummy, 
     &              huge(NPARTSASS),  
     &              INFO_TMP)
        IFLAG  = INFO_TMP(1) 
        IERROR = INFO_TMP(2) 
        IF (IFLAG.LT.0) GOTO 500
      ENDIF
      LASTPANEL = .FALSE.
      DO WHILE (IEND_BLR < NASS ) 
        CURRENT_BLR = CURRENT_BLR + 1
        IBEG_BLR = IW(IOLDPS+1+KEEP(IXSZ)) + 1 
        IF (.NOT. LR_ACTIVATED) THEN
          IEND_BLR = min(IEND_BLR + NBLR_ORIG, NASS)
        ELSE
          IEND_BLR = BEGS_BLR(CURRENT_BLR+1)-1
          BEGS_BLR( CURRENT_BLR ) = IBEG_BLR
          IF ( IEND_BLR - IBEG_BLR + 1 .GT. MAXI_CLUSTER ) THEN
            MAXI_CLUSTER = IEND_BLR - IBEG_BLR + 1
            LWORK = MAXI_CLUSTER*(MAXI_CLUSTER+1)
            DEALLOCATE(BLOCK, WORK, RWORK, TAU, JPVT)
            ALLOCATE(BLOCK(MAXI_CLUSTER, OMP_NUM*MAXI_CLUSTER),
     &             RWORK(2*MAXI_CLUSTER*OMP_NUM), 
     &             TAU(MAXI_CLUSTER*OMP_NUM),
     &             JPVT(MAXI_CLUSTER*OMP_NUM), 
     &             WORK(LWORK*OMP_NUM),stat=allocok)
            IF (allocok > 0) THEN
              IFLAG  = -13
              IERROR = OMP_NUM*(LWORK + MAXI_CLUSTER*(MAXI_CLUSTER+4))
              GOTO 480
            ENDIF
            IF (KEEP(480).GE.3) THEN
              DO MY_NUM=1,OMP_NUM
                CALL DEALLOC_LRB(ACC_LUA(MY_NUM), KEEP8, KEEP(34))
                CALL ALLOC_LRB(ACC_LUA(MY_NUM), MAXI_RANK,
     &                         MAXI_CLUSTER, MAXI_CLUSTER, .TRUE.,
     &                         IFLAG, IERROR, KEEP8)
                IF (IFLAG.LT.0) GOTO 480
                ACC_LUA(MY_NUM)%K = 0
              ENDDO
            ENDIF
          ENDIF
        ENDIF
        DO WHILE (IEND_BLOCK < IEND_BLR ) 
          IBEG_BLOCK = IW(IOLDPS+1+KEEP(IXSZ)) + 1
          IF (KEEP(405).EQ.0) THEN
            KEEP(425)=max(KEEP(425),IEND_BLOCK-IBEG_BLOCK)
          ELSE
!$OMP       ATOMIC UPDATE
            KEEP(425)=max(KEEP(425),IEND_BLOCK-IBEG_BLOCK)
!$OMP       END ATOMIC
          ENDIF
          IEND_BLOCK = min(IEND_BLOCK + NBKJIB_ORIG, IEND_BLR)
  50      CONTINUE  
            IF (K263.EQ.0) THEN
              IBEG_BLOCK_FOR_IPIV = IBEG_BLOCK
            ELSE
              IBEG_BLOCK_FOR_IPIV = IBEG_BLR
            ENDIF
            CALL DMUMPS_FAC_I_LDLT_NIV2(
     &                DIAG_ORIG, SIZEDIAG_ORIG, GW_FACTCUMUL,
     &                NFRONT,NASS,IBEG_BLOCK_FOR_IPIV,
     &                IBEG_BLOCK, IEND_BLOCK,
     &                NASS, IPIV,
     &                N,INODE,IW,LIW,A,LA,
     &                NNEGW,NNULLNEGW, NB22T2W,NBTINYW,
     &                DET_EXPW, DET_MANTW, DET_SIGNW,
     &                INOPV,
     &                IFLAG,IERROR,IOLDPS,POSELT,UU, SEUIL_LOC,
     &                KEEP,KEEP8,PIVSIZ,
     &           DKEEP(1), PIVNUL_LIST_STRUCT, SWAP_OCCURRED,
     &           PP_FIRST2SWAP_L, MonBloc%LastPanelWritten_L,
     &           PP_LastPIVRPTRFilled,
     &           PIVOT_OPTION,
     &           Inextpiv, IEND_BLR, LR_ACTIVATED, 
     &           OOC_EFFECTIVE_ON_FRONT)
            IF (IFLAG.LT.0) GOTO 500
            IF (INOPV.EQ. 1) THEN
              IF (STATICMODE) THEN
                INOPV = -1
                GOTO 50
              ENDIF
             LASTPANEL = .TRUE.
            ELSE IF (INOPV .LE. 0) THEN 
              INOPV = 0
              NPVW = NPVW + PIVSIZ
              CALL DMUMPS_FAC_MQ_LDLT_NIV2(IEND_BLOCK,
     &             NASS, IW(IOLDPS+1+XSIZE), INODE,A,LA,
     &             LDAFS, POSELT,IFINB,
     &             PIVSIZ,
     &             KEEP(219),
     &             PIVOT_OPTION, IEND_BLR, LR_ACTIVATED)
              IF(PIVSIZ .EQ. 2) THEN
                IWPOSPIV = IOLDPS+XSIZE+IW(IOLDPS+1+XSIZE)+6+
     &                     IW(IOLDPS+5+XSIZE)
                IW(IWPOSPIV+NFRONT) = -IW(IWPOSPIV+NFRONT)
              ENDIF
              IW(IOLDPS+1+XSIZE) = IW(IOLDPS+1+XSIZE) + PIVSIZ
            IF (IFINB.EQ.0) THEN
              GOTO 50 
            ELSE IF (IFINB .EQ. -1) THEN
              LASTPANEL = .TRUE.
            ENDIF
          ENDIF
          NPIV = IW(IOLDPS+1+XSIZE)
          IF ( OOC_EFF_AND_WRITE_BYPANEL ) THEN
            IF (.NOT.RESET_TO_ONE.OR.K109_SAVE.EQ.KEEP(109)) THEN
              MonBloc%Last   = .FALSE.
              MonBloc%LastPiv= NPIV
              LAST_CALL=.FALSE.
              CALL DMUMPS_OOC_IO_LU_PANEL(
     &        STRAT_TRY_WRITE,
     &        TYPEFile, A(POSELT),
     &        LAFAC, MonBloc, NextPiv2beWritten, IDUMMY, IW(IOLDPS),
     &        LIWFAC, MYID, KEEP8(31), IFLAG_OOC,LAST_CALL )
              IF (IFLAG_OOC .LT. 0 ) THEN
                IFLAG = IFLAG_OOC
                GOTO 500
              ENDIF
            ENDIF
          ENDIF
          IF (K263.eq.0) THEN
            NELIM = IEND_BLR - NPIV
            CALL DMUMPS_SEND_FACTORED_PANEL( COMM_LOAD, ASS_IRECV, 
     &             N, INODE, FPERE, IW, LIW, 
     &             IOLDPS, POSELT, A, LA, LDAFS,
     &             IBEG_BLOCK, NPIV, IPIV, NASS,LASTPANEL, NB_BLOC_FAC,
     &             COMM, MYID, BUFR, LBUFR, LBUFR_BYTES,NBFIN,LEAF,
     &             IFLAG, IERROR, IPOOL,LPOOL,
     &             SLAVEF, POSFAC, IWPOS, IWPOSCB, IPTRLU, LRLU,
     &             LRLUS, COMP, PTRIST, PTRAST, PTLUST_S, PTRFAC, STEP,
     &             PIMASTER, PAMASTER,
     &             NSTK_S,PERM,PROCNODE_STEPS, root, roota,
     &             OPASSW, OPELIW, ITLOC, RHS_MUMPS,
     &             FILS, DAD, PTRARW, PTRAIW,
     &             PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &             INTARR,DBLARR,ICNTL,KEEP,KEEP8,DKEEP,ND,FRERE,
     &             LPTRAR, NELT, FRTPTR, FRTELT, 
     &             ISTEP_TO_INIV2, TAB_POS_IN_PERE
     &             , NELIM, .FALSE. 
     &             , NPARTSASS, CURRENT_BLR, BLR_DUMMY, LRGROUPS
     & )
            IF ( IFLAG .LT. 0 ) GOTO 500
            IF (RESET_TO_ONE.AND.K109_SAVE.LT.KEEP(109)) THEN
              CALL DMUMPS_RESET_TO_ONE( 
     &        IW(IOLDPS+KEEP(IXSZ)+IW(IOLDPS+5+KEEP(IXSZ))+6),
     &        NPIV, IBEG_BLOCK,
     &        K109_SAVE, KEEP(109), PIVNUL_LIST_STRUCT,
     &        A, POSELT, LA, LDAFS)
            ENDIF
            IF ( OOC_EFF_AND_WRITE_BYPANEL) THEN
              MonBloc%Last  = .FALSE.
              MonBloc%LastPiv= NPIV
              LAST_CALL=.FALSE.
              CALL DMUMPS_OOC_IO_LU_PANEL(
     &        STRAT_TRY_WRITE,
     &        TYPEFile, A(POSELT),
     &        LAFAC, MonBloc, NextPiv2beWritten, IDUMMY, IW(IOLDPS),
     &        LIWFAC, MYID, KEEP8(31), IFLAG_OOC,LAST_CALL )
              IF (IFLAG_OOC .LT. 0 ) THEN
                IFLAG = IFLAG_OOC
                GOTO 500
              ENDIF
            ENDIF
          ENDIF
          IF ( IEND_BLR .GT. IEND_BLOCK ) THEN
              IF (PIVOT_OPTION.EQ.2) THEN
                LAST_ROW = NASS
              ELSE
                LAST_ROW = IEND_BLR
              ENDIF
              CALL DMUMPS_FAC_SQ_LDLT(IBEG_BLOCK,IEND_BLOCK,NPIV,
     &             NASS,NASS,INODE,A,LA,
     &             LDAFS, POSELT,
     &             KEEP,KEEP8,
     &             -6666, -6666, 
     &             IEND_BLR, LAST_ROW,
     &             .FALSE., .TRUE., LR_ACTIVATED,
     &             IW, LIW, -6666 
     &             )
          ENDIF
          CALL MUMPS_BUF_TEST()
        END DO 
        NPIV   = IW(IOLDPS+1+XSIZE)
        IF (LR_ACTIVATED) THEN
          ALLOCATE(BLR_L(NB_BLR-CURRENT_BLR),stat=allocok)
          IF (allocok > 0) THEN
             IFLAG  = -13
             IERROR = NB_BLR-CURRENT_BLR
             GOTO 500
          ENDIF
          NELIM = IEND_BLOCK - NPIV
          IF (IEND_BLR.NE.IEND_BLOCK) THEN
            WRITE(*,*) "Internal error 1 in DMUMPS_FAC2_LDLT",
     &      IEND_BLR, IEND_BLOCK
            CALL MUMPS_ABORT()
          ENDIF
          IF (NELIM .EQ. IEND_BLR - IBEG_BLR + 1) THEN
            IF (KEEP(480).GE.2
     &       .OR.
     &       (
     &         (KEEP(486).EQ.2) 
     &       )
     &         ) THEN
              DO J=1,NB_BLR-CURRENT_BLR
                 BLR_L(J)%M=0
                 BLR_L(J)%N=0
                 BLR_L(J)%K=0
                 BLR_L(J)%ISLR=.FALSE.
                 NULLIFY(BLR_L(J)%Q)
                 NULLIFY(BLR_L(J)%R)
              ENDDO
              CALL DMUMPS_BLR_SAVE_PANEL_LORU (
     &              IW(IOLDPS+XXF),
     &              0, 
     &        CURRENT_BLR, BLR_L, 0) 
              NULLIFY(BLR_L)
            ENDIF
            GOTO 101
          ENDIF
#if ! defined(BLR_NOOPENMP)
!$OMP PARALLEL 
#endif
          CALL DMUMPS_COMPRESS_PANEL(A, LA, POSELT, IFLAG, IERROR, NASS,
     &         BEGS_BLR, NB_BLR, DKEEP(8), KEEP(466), KEEP(458),
     &         KEEP(473), BLR_L, 
     &         CURRENT_BLR, 'V', WORK, TAU, JPVT, LWORK, RWORK,
     &         BLOCK, MAXI_CLUSTER, NELIM, 
     &         .FALSE., 0, 0,
     &         2, KEEP(483), KEEP8
     &        )
#if ! defined(BLR_NOOPENMP)
!$OMP BARRIER
#endif          
          IF (IFLAG.LT.0) GOTO 400
          IF (PIVOT_OPTION.LT.2) THEN
            CALL DMUMPS_BLR_PANEL_LRTRSM(A, LA, POSELT, NFRONT,
     &                IBEG_BLR,
     &                NB_BLR, BLR_L, CURRENT_BLR, CURRENT_BLR+1, 
     &                NB_BLR, 2, 1, 0, .FALSE.,
     &                IW, OFFSET_IW=IOLDPS+6+XSIZE+NFRONT+IBEG_BLR-1,
     &                NASS=NASS)     
#if ! defined(BLR_NOOPENMP)
!$OMP BARRIER
#endif          
          ENDIF
 400      CONTINUE         
#if ! defined(BLR_NOOPENMP)
!$OMP END PARALLEL
#endif          
          IF (IFLAG.LT.0) GOTO 480
          IF (KEEP(480).NE.0
     &       .OR.
     &       (
     &         (KEEP(486).EQ.2) 
     &       )
     &       ) THEN
            IF (KEEP(480).LT.5) THEN
              CALL DMUMPS_BLR_SAVE_PANEL_LORU (
     &              IW(IOLDPS+XXF),
     &              0, 
     &        CURRENT_BLR, BLR_L, 0) 
            ENDIF
          ENDIF
        ENDIF
 101    CONTINUE       
        IF (.NOT. LR_ACTIVATED) THEN
          CALL DMUMPS_FAC_SQ_LDLT(IBEG_BLR,IEND_BLR,NPIV,
     &             NASS, NASS, INODE, A, LA,
     &             LDAFS, POSELT,
     &             KEEP, KEEP8,
     &             IEND_BLR, NASS,
     &             -6666, -6666, 
     &             (PIVOT_OPTION.LE.1), .FALSE., LR_ACTIVATED,
     &             IW, LIW, IOLDPS+6+XSIZE+NFRONT+IBEG_BLR-1) 
        ENDIF
        IF (K263.NE.0) THEN
          NELIM = IEND_BLR-NPIV
          BLR_SEND=>BLR_DUMMY
          IF (associated(BLR_L)) THEN
            BLR_SEND=>BLR_L
          ENDIF
          CALL DMUMPS_SEND_FACTORED_PANEL( COMM_LOAD, ASS_IRECV,
     &             N, INODE, FPERE, IW, LIW, 
     &             IOLDPS, POSELT, A, LA, LDAFS,
     &             IBEG_BLR, NPIV, IPIV, NASS,LASTPANEL, NB_BLOC_FAC,
     &             COMM, MYID, BUFR, LBUFR, LBUFR_BYTES,NBFIN,LEAF,
     &             IFLAG, IERROR, IPOOL,LPOOL,
     &             SLAVEF, POSFAC, IWPOS, IWPOSCB, IPTRLU, LRLU,
     &             LRLUS, COMP, PTRIST, PTRAST, PTLUST_S,PTRFAC,STEP,
     &             PIMASTER, PAMASTER,
     &             NSTK_S,PERM,PROCNODE_STEPS, root, roota,
     &             OPASSW, OPELIW, ITLOC, RHS_MUMPS,
     &             FILS, DAD, PTRARW, PTRAIW,
     &             PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &             INTARR,DBLARR,ICNTL,KEEP,KEEP8,DKEEP,ND,FRERE,
     &             LPTRAR, NELT, FRTPTR, FRTELT, 
     &             ISTEP_TO_INIV2, TAB_POS_IN_PERE
     &             , NELIM, LR_ACTIVATED
     &             , NPARTSASS, CURRENT_BLR , BLR_SEND , LRGROUPS
     &             )
          IF ( IFLAG .LT. 0 ) GOTO 500
          IF (RESET_TO_ONE.AND.K109_SAVE.LT.KEEP(109)) THEN
              CALL DMUMPS_RESET_TO_ONE( 
     &        IW(IOLDPS+KEEP(IXSZ)+IW(IOLDPS+5+KEEP(IXSZ))+6),
     &        NPIV, IBEG_BLR,
     &        K109_SAVE, KEEP(109), PIVNUL_LIST_STRUCT,
     &        A, POSELT, LA, LDAFS)
          ENDIF
          IF ( OOC_EFF_AND_WRITE_BYPANEL ) THEN
              MonBloc%Last  = .FALSE.
              MonBloc%LastPiv= NPIV
              LAST_CALL=.FALSE.
              CALL DMUMPS_OOC_IO_LU_PANEL(
     &        STRAT_TRY_WRITE,
     &        TYPEFile, A(POSELT),
     &        LAFAC, MonBloc, NextPiv2beWritten, IDUMMY, IW(IOLDPS),
     &        LIWFAC, MYID, KEEP8(31), IFLAG_OOC,LAST_CALL )
              IF (IFLAG_OOC .LT. 0 ) THEN
                IFLAG = IFLAG_OOC
                GOTO 500
              ENDIF
          ENDIF
        ENDIF
        IF (.NOT. LR_ACTIVATED) THEN
          IF (PIVOT_OPTION.EQ.2) THEN
            LAST_ROW = NASS
          ELSE
            LAST_ROW = IEND_BLR
          ENDIF
          CALL DMUMPS_FAC_SQ_LDLT(IBEG_BLR,IEND_BLR,NPIV,
     &             NASS,NASS,INODE,A,LA,
     &             LDAFS, POSELT,
     &             KEEP,KEEP8,
     &             -6666, -6666, 
     &             NASS, LAST_ROW, 
     &             .FALSE., .TRUE., LR_ACTIVATED,
     &             IW, LIW, -6666 
     &             )
        ELSE
          NELIM = IEND_BLOCK - NPIV
          IF (IEND_BLR.NE.IEND_BLOCK) THEN
             CALL MUMPS_ABORT()
          ENDIF
#if ! defined(BLR_NOOPENMP)
!$OMP PARALLEL
#endif
          IF (KEEP(480).GE.2) THEN
            IF (IEND_BLR.LT.NASS) THEN
              CALL DMUMPS_BLR_UPD_PANEL_LEFT_LDLT(A, LA, POSELT,
     &          NASS, IW(IOLDPS+XXF), 
     &          BEGS_BLR, CURRENT_BLR, NB_BLR, NPARTSASS,
     &          NELIM,
     &          IW(HF+IOLDPS+NFRONT), BLOCK,
     &          ACC_LUA, MAXI_CLUSTER, MAXI_RANK,
     &          2, IFLAG, IERROR,
     &          KEEP(481), DKEEP(11), KEEP(466), KEEP(477), 
     &          KEEP(480), KEEP(479), KEEP(478), KEEP(476), 
     &          KEEP(483), KEEP8)
            ENDIF
          ENDIF
          IF (NELIM .EQ. IEND_BLR - IBEG_BLR + 1) GOTO 450
          IF (KEEP(480).LT.2) THEN
            CALL DMUMPS_BLR_UPDATE_TRAILING_LDLT(A, LA, POSELT, 
     &        IFLAG, IERROR, NASS,
     &        BEGS_BLR, NB_BLR, CURRENT_BLR, BLR_L, NELIM,
     &        IW(HF+IOLDPS+NFRONT+IBEG_BLR-1), BLOCK,
     &        MAXI_CLUSTER, NPIV,
     &        2, 
     &        KEEP(481), DKEEP(11), KEEP(466), KEEP(477) 
     &        )
          ENDIF
#if ! defined(BLR_NOOPENMP)
!$OMP BARRIER
#endif          
            IF (IFLAG.LT.0) GOTO 450
          IF (PIVOT_OPTION.LT.2) THEN
            IF ((UU.GT.0).OR.(KEEP(486).NE.2)) THEN
              CALL DMUMPS_DECOMPRESS_PANEL(A, LA, POSELT, NASS, NASS,
     &              .TRUE.,  
     &         BEGS_BLR(CURRENT_BLR),
     &         BEGS_BLR(CURRENT_BLR+1), NB_BLR, BLR_L, CURRENT_BLR, 
     &         'V', 1)
            ENDIF
          ENDIF
 450      CONTINUE
#if ! defined(BLR_NOOPENMP)
!$OMP END PARALLEL
#endif          
          IF (IFLAG.LT.0) GOTO 480
          IF (NELIM .EQ. IEND_BLR - IBEG_BLR + 1) THEN
            IF (KEEP(486).EQ.3) THEN
              IF (KEEP(480).EQ.0) THEN
                DEALLOCATE(BLR_L)
                NULLIFY(BLR_L)
              ENDIF
            ENDIF
            GOTO 100
          ENDIF
          IF (KEEP(486).EQ.3) THEN
            IF (KEEP(480).EQ.0) THEN
              CALL DEALLOC_BLR_PANEL(BLR_L, NB_BLR-CURRENT_BLR, KEEP8,
     &                               KEEP(34))
              DEALLOCATE(BLR_L)
            ENDIF
            NULLIFY(BLR_L)
          ENDIF
        ENDIF 
        IF ( OOC_EFF_AND_WRITE_BYPANEL ) THEN
          MonBloc%Last   = .FALSE.
          MonBloc%LastPiv= NPIV
          LAST_CALL=.FALSE.
          CALL DMUMPS_OOC_IO_LU_PANEL(
     &        STRAT_TRY_WRITE,
     &        TYPEFile, A(POSELT),
     &        LAFAC, MonBloc, NextPiv2beWritten, IDUMMY, IW(IOLDPS),
     &        LIWFAC, MYID, KEEP8(31), IFLAG_OOC,LAST_CALL )
              IF (IFLAG_OOC .LT. 0 ) THEN
                IFLAG = IFLAG_OOC
                GOTO 500
              ENDIF
        ENDIF
  100   CONTINUE
      END DO 
      IF (LR_ACTIVATED) THEN
        IBEG_BLR = IW(IOLDPS+1+XSIZE) + 1 
        BEGS_BLR( CURRENT_BLR + 1 ) = IBEG_BLR
        IF ( (KEEP(486).EQ.2) 
     &       ) THEN
          CALL DMUMPS_BLR_RETRIEVE_BEGSBLR_STA(IW(IOLDPS+XXF),
     &                      BEGS_BLR_STATIC)
          IF (UU.GT.0 .AND. SWAP_OCCURRED) THEN
            allocate(BEGS_BLR_TMP(NB_BLR+1),stat=allocok)
            IF (allocok > 0) THEN
              IFLAG  = -13
              IERROR = NB_BLR+1
              GOTO 500
            ENDIF
            DO J=1,NB_BLR+1
               BEGS_BLR_TMP(J) = BEGS_BLR_STATIC(J)
            ENDDO
          ENDIF
        ENDIF
        IF (
     &         (KEEP(486).EQ.2) 
     &       ) THEN
          MEM_TOT = 0
#if ! defined(BLR_NOOPENMP)
!$OMP PARALLEL
!$OMP& PRIVATE(IP, NELIM)
#endif
#if ! defined(BLR_NOOPENMP)
!$OMP DO PRIVATE(DIAG, DIAGSIZ_STA, DIAGSIZ_DYN, DPOS, POSELT_DIAG, MEM,
!$OMP&           allocok)
!$OMP&   REDUCTION(+:MEM_TOT)
#endif
          DO IP=1,NPARTSASS
            IF (IFLAG.LT.0) CYCLE
            DIAGSIZ_DYN = BEGS_BLR(IP+1)-BEGS_BLR(IP)
            DIAGSIZ_STA = BEGS_BLR_STATIC(IP+1)-BEGS_BLR(IP)
            MEM = DIAGSIZ_DYN*DIAGSIZ_STA
            MEM_TOT = MEM_TOT + MEM
            ALLOCATE(DIAG(MEM), stat=allocok)
            IF (allocok > 0) THEN
              IFLAG  = -13
              IERROR = MEM
              CYCLE
            ENDIF 
            DPOS = 1
            POSELT_DIAG = POSELT + int(BEGS_BLR(IP)-1,8)*int(LDAFS,8)
     &                         + int(BEGS_BLR(IP)-1,8)
            DO I=1,DIAGSIZ_STA
              DIAG(DPOS:DPOS+DIAGSIZ_DYN-1) =
     &               A(POSELT_DIAG:POSELT_DIAG+int(DIAGSIZ_DYN-1,8))
              DPOS = DPOS + DIAGSIZ_DYN
              POSELT_DIAG = POSELT_DIAG + int(LDAFS,8)
            ENDDO
            CALL DMUMPS_BLR_SAVE_DIAG_BLOCK(
     &            IW(IOLDPS+XXF),
     &            IP, DIAG, KEEP(34))
          ENDDO
#if ! defined(BLR_NOOPENMP)
!$OMP ENDDO
!$OMP SINGLE
#endif
          CALL MUMPS_DM_FAC_UPD_DYN_MEMCNTS(int(MEM_TOT,8),
     &          .FALSE., KEEP8, IFLAG, IERROR, .TRUE., .TRUE.)
#if ! defined(BLR_NOOPENMP)
!$OMP END SINGLE
#endif
          IF (IFLAG.LT.0) GOTO 460
          IF (UU.GT.0 .AND. SWAP_OCCURRED) THEN
            DO IP=1,NPARTSASS
              NELIM = BEGS_BLR_TMP(IP+1)-BEGS_BLR(IP+1)
#if ! defined(BLR_NOOPENMP)
!$OMP SINGLE
#endif
              CALL DMUMPS_BLR_RETRIEVE_PANEL_LORU(
     &        IW(IOLDPS+XXF), 0, IP, BLR_PANEL)
              CALL DEALLOC_BLR_PANEL(BLR_PANEL, NPARTSASS-IP, KEEP8,
     &                             KEEP(34))
#if ! defined(BLR_NOOPENMP)
!$OMP END SINGLE
#endif
              CALL DMUMPS_COMPRESS_PANEL(A, LA, POSELT, IFLAG,
     &        IERROR, LDAFS, BEGS_BLR_TMP,
     &        NB_BLR, DKEEP(8), KEEP(466), KEEP(458), KEEP(473),
     &        BLR_PANEL, IP,
     &        'V', WORK, TAU, JPVT, LWORK, RWORK,
     &        BLOCK, MAXI_CLUSTER, NELIM,
     &        .FALSE., 0, 0,
     &        2, KEEP(483), KEEP8,
     &        END_I_IN=NPARTSASS, FRSWAP=.TRUE.
     &        )
#if ! defined(BLR_NOOPENMP)
!$OMP BARRIER
#endif
              IF (IFLAG.LT.0) GOTO 440
#if ! defined(BLR_NOOPENMP)
!$OMP SINGLE
#endif
              BEGS_BLR_TMP(IP+1) = BEGS_BLR(IP+1)
#if ! defined(BLR_NOOPENMP)
!$OMP END SINGLE
#endif
            ENDDO
#if ! defined(BLR_NOOPENMP)
!$OMP BARRIER
#endif
 440        CONTINUE
          ENDIF 
 460      CONTINUE          
#if ! defined(BLR_NOOPENMP)
!$OMP END PARALLEL
#endif
          IF (UU.GT.0 .AND. SWAP_OCCURRED) THEN
            deallocate(BEGS_BLR_TMP)
          ENDIF
          IF (IFLAG.LT.0) GOTO 500
        ENDIF 
        IF (
     &         (KEEP(486).EQ.2) 
     &       ) THEN
            CALL DMUMPS_BLR_SAVE_BEGS_BLR_DYN(IW(IOLDPS+XXF),
     &        BEGS_BLR)
        ENDIF
      ENDIF 
      IF ( OOC_EFFECTIVE_ON_FRONT ) THEN
          STRAT        = STRAT_WRITE_MAX   
          MonBloc%Last = .TRUE.
          MonBloc%LastPiv  = IW(IOLDPS+1+XSIZE)
          LAST_CALL    = .TRUE.
          CALL DMUMPS_OOC_IO_LU_PANEL
     &          ( STRAT, TYPEFile, 
     &           A(POSELT), LAFAC, MonBloc,
     &           NextPiv2beWritten, IDUMMY,
     &           IW(IOLDPS), LIWFAC, 
     &           MYID, KEEP8(31), IFLAG_OOC, LAST_CALL )
          IF (IFLAG_OOC .LT. 0 ) THEN
            IFLAG = IFLAG_OOC
            GOTO 500
          ENDIF
          CALL DMUMPS_OOC_PP_TRYRELEASE_SPACE (IWPOS, 
     &      IOLDPS, IW, LIW, MonBloc , NFRONT, KEEP)
      ENDIF
      GOTO 600
 480  CONTINUE
 500  CONTINUE
      CALL DMUMPS_BDC_ERROR( MYID, SLAVEF, COMM, KEEP )
 600  CONTINUE
      IF(allocated(IPIV)) DEALLOCATE( IPIV )
      IF (allocated(DIAG_ORIG)) DEALLOCATE(DIAG_ORIG)
      IF (LR_ACTIVATED) THEN
         IF (IFLAG.GE.0) THEN
           CALL UPD_MRY_LU_FR(NASS, NFRONT-NASS, 1, NELIM)
           DO IP=1,NPARTSASS
             CALL DMUMPS_BLR_RETRIEVE_PANEL_LORU(
     &               IW(IOLDPS+XXF), 0, IP, BLR_PANEL)
             CALL UPD_MRY_LU_LRGAIN(BLR_PANEL, NPARTSASS-IP
     &            )
           ENDDO
           CALL UPD_FLOP_FACTO_FR(NFRONT, NASS, NASS-NELIM, 2, 2)
         ENDIF 
         IF (allocated(RWORK)) DEALLOCATE(RWORK)
         IF (allocated(WORK))  DEALLOCATE(WORK)
         IF (allocated(TAU))   DEALLOCATE(TAU)
         IF (allocated(JPVT))  DEALLOCATE(JPVT)
         IF (allocated(BLOCK)) DEALLOCATE(BLOCK)
         IF (associated(ACC_LUA)) THEN 
           IF (KEEP(480).GE.3) THEN
             DO MY_NUM=1,OMP_NUM
               CALL DEALLOC_LRB(ACC_LUA(MY_NUM), KEEP8, KEEP(34))
             ENDDO
           ENDIF
           DEALLOCATE(ACC_LUA)
         ENDIF
         IF (associated(BEGS_BLR)) THEN
           DEALLOCATE(BEGS_BLR)
           NULLIFY(BEGS_BLR)
         ENDIF
      ENDIF
      IF (KEEP(486).NE.0) THEN
        IF (.NOT.LR_ACTIVATED) THEN
          CALL UPD_FLOP_FRFRONTS(NFRONT, NPIV, NASS, KEEP(50), 2)
        ENDIF
      ENDIF
      IF (LR_ACTIVATED.AND.KEEP(480).NE.0) THEN
        IF (.NOT.
     &       (
     &         (KEEP(486).EQ.2) 
     &       )
     &     ) THEN
          CALL DMUMPS_BLR_FREE_ALL_PANELS(IW(IOLDPS+XXF), 0, 
     &                    KEEP8, KEEP(34))
        ENDIF
      ENDIF
      IF (LR_ACTIVATED) THEN
        IF (.NOT.
     &       (
     &         (KEEP(486).EQ.2) 
     &       )
     &     ) THEN
          CALL DMUMPS_BLR_END_FRONT(IW(IOLDPS+XXF), IFLAG, KEEP8,
     &                              KEEP(34))
        ENDIF
      ENDIF
      RETURN
      END SUBROUTINE DMUMPS_FAC2_LDLT
      SUBROUTINE DMUMPS_RESET_TO_ONE(FRONT_INDEX_LIST, NPIV,
     & IBEG_BLOCK, K109_SAVE, K109, PIVNUL_LIST_STRUCT,
     & A, POSELT, LA, LDAFS)
      USE MUMPS_PIVNUL_MOD, ONLY: PIVNUL_LIST_STRUCT_T
      INTEGER, INTENT(IN) :: NPIV, IBEG_BLOCK
      INTEGER, INTENT(IN) :: FRONT_INDEX_LIST(NPIV)
      INTEGER, INTENT(IN) :: K109
      INTEGER, INTENT(INOUT) :: K109_SAVE
      TYPE(PIVNUL_LIST_STRUCT_T)     :: PIVNUL_LIST_STRUCT
      INTEGER(8), INTENT(IN) :: POSELT, LA
      INTEGER, INTENT(IN) :: LDAFS
      DOUBLE PRECISION, INTENT(INOUT) :: A(LA)
      LOGICAL :: TO_UPDATE
      INTEGER :: I, JJ, K
      DOUBLE PRECISION ONE
      PARAMETER (ONE = 1.0D0)
      DO K = K109_SAVE+1, K109
        TO_UPDATE = .FALSE. 
        I = PIVNUL_LIST_STRUCT%PIVNUL_LIST(K)  
        DO JJ=IBEG_BLOCK, NPIV
          IF (FRONT_INDEX_LIST(JJ) .EQ.I) THEN
            TO_UPDATE=.TRUE. 
            EXIT
          ENDIF
        ENDDO
        IF (TO_UPDATE) THEN
          A(POSELT+int(JJ,8)+int(LDAFS,8)*int(JJ-1,8))= ONE
          TO_UPDATE=.FALSE. 
       ELSE
          write(*,*) ' Internal error related ', 
     &               'to null pivot row detection'
          CALL MUMPS_ABORT()
        ENDIF
      ENDDO
      K109_SAVE = K109
      RETURN
      END SUBROUTINE DMUMPS_RESET_TO_ONE
      END MODULE DMUMPS_FAC2_LDLT_M
