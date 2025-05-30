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
      MODULE DMUMPS_SOL_LR
      USE DMUMPS_LR_TYPE
      USE DMUMPS_LR_CORE
      USE MUMPS_LR_STATS
      USE DMUMPS_LR_DATA_M, only: BLR_ARRAY
      IMPLICIT NONE
      CONTAINS
      SUBROUTINE DMUMPS_SOL_FWD_LR_SU 
     &    (INODE, N, IWHDLR, NPIV_GLOBAL, NSLAVES,
     &     IW, IPOS_INIT, LIW, 
     &     LIELL, WCB, LWCB, 
     &     LD_WCBPIV, LD_WCBCB,
     &     PPIV_INIT, PCB_INIT, 
     &     RHSINTR, LRHSINTR, NRHS, 
     &     POSINRHSINTR_FWD, JBDEB, JBFIN, 
     &     MTYPE, KEEP, KEEP8, OOCWRITE_COMPATIBLE_WITH_BLR,
     &     IFLAG, IERROR )
       INTEGER, INTENT(IN) :: INODE, N, IWHDLR, NPIV_GLOBAL, NSLAVES
       INTEGER, INTENT(IN) :: MTYPE, LIELL, KEEP(500)
       INTEGER(8), INTENT(IN) :: KEEP8(150)
       INTEGER, INTENT(IN) :: LIW, IPOS_INIT, LRHSINTR
       INTEGER, INTENT(IN) :: IW(LIW), POSINRHSINTR_FWD(N)
       INTEGER(8), INTENT(IN) :: LWCB, PPIV_INIT, PCB_INIT
       INTEGER, INTENT(IN) :: LD_WCBPIV, LD_WCBCB, NRHS, JBDEB, JBFIN
       DOUBLE PRECISION, INTENT(INOUT) :: WCB(LWCB)
       INTEGER, INTENT(INOUT) :: IFLAG, IERROR
       DOUBLE PRECISION, INTENT(INOUT) :: RHSINTR(LRHSINTR, NRHS)
       LOGICAL, INTENT(IN)    :: OOCWRITE_COMPATIBLE_WITH_BLR
        INTEGER :: I, NPARTSASS, NB_BLR , NELIM, LDADIAG,
     &             DIAGSIZ_DYN, DIAGSIZ_STA, IBEG_BLR, IEND_BLR,
     &             LD_CB, NRHS_B, IPOS, KCB
        INTEGER(8) :: PPIV, PCB
        INTEGER    :: LAST_BLR
        DOUBLE PRECISION, POINTER, DIMENSION(:) :: DIAG
        TYPE(LRB_TYPE), POINTER, DIMENSION(:) :: BLR_PANEL
      DOUBLE PRECISION :: ONE, MONE, ZERO
      PARAMETER (ONE = 1.0D0, MONE=-1.0D0)
      PARAMETER (ZERO=0.0D0)
        NRHS_B = JBFIN-JBDEB+1
        IF (MTYPE.EQ.1) THEN
         IF (associated(BLR_ARRAY(IWHDLR)%PANELS_L))
     &      THEN
            NPARTSASS=size(BLR_ARRAY(IWHDLR)%PANELS_L)
            NB_BLR = size(BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC) -1
         ELSE
           WRITE(6,*) " Internal error in DMUMPS_SOL_FWD_SU_MASTER"
         ENDIF
        ELSE
         IF (associated(BLR_ARRAY(IWHDLR)%PANELS_U))
     &      THEN
            NPARTSASS=size(BLR_ARRAY(IWHDLR)%PANELS_U)
            NB_BLR = size(BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC) -1
         ENDIF
        ENDIF
        IF (NSLAVES.EQ.0 .OR. (KEEP(50).eq.0 .and. MTYPE .NE.1)) THEN
          LAST_BLR = NB_BLR
        ELSE
          LAST_BLR = NPARTSASS
        ENDIF
      IPOS = IPOS_INIT
      PPIV = PPIV_INIT 
      DO I=1, NPARTSASS
        IBEG_BLR = BLR_ARRAY(IWHDLR)%BEGS_BLR_DYNAMIC(I)
        IEND_BLR = BLR_ARRAY(IWHDLR)%BEGS_BLR_DYNAMIC(I+1) -1
        DIAGSIZ_DYN = BLR_ARRAY(IWHDLR)%BEGS_BLR_DYNAMIC(I+1) -
     &           IBEG_BLR
        DIAGSIZ_STA = BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC(I+1)  - 
     &           IBEG_BLR 
        IF (KEEP(50).NE.0) THEN
         LDADIAG = DIAGSIZ_DYN
        ELSE
         LDADIAG = DIAGSIZ_STA
        ENDIF 
        IF (IEND_BLR.EQ.NPIV_GLOBAL) THEN
          PCB   = PCB_INIT
        ELSE
          PCB  = PPIV + int(DIAGSIZ_DYN,8)
        ENDIF
        IF ( DIAGSIZ_DYN.EQ.0) CYCLE
        NELIM = DIAGSIZ_STA - DIAGSIZ_DYN
         IF ( MTYPE .EQ. 1 ) THEN
          BLR_PANEL => BLR_ARRAY(IWHDLR)%PANELS_L(I)%LRB_PANEL
         ELSE
          BLR_PANEL => BLR_ARRAY(IWHDLR)%PANELS_U(I)%LRB_PANEL
         END IF
         DIAG => BLR_ARRAY(IWHDLR)%DIAG_BLOCKS(I)%DIAG_BLOCK 
         CALL DMUMPS_SOLVE_FWD_TRSOLVE (DIAG(1), 
     &         int(size(DIAG),8), 1_8,
     &         DIAGSIZ_DYN , LDADIAG, NRHS_B, WCB, LWCB, NPIV_GLOBAL,
     &         PPIV, MTYPE, KEEP)
         IF (NELIM.GT.0) THEN
           KCB = int(PCB-PPIV_INIT+1)
           IF (IEND_BLR.EQ.NPIV_GLOBAL) THEN 
             LD_CB = LD_WCBCB
           ELSE
             LD_CB = LD_WCBPIV
           ENDIF
           IF (MTYPE.EQ.1) THEN
             IF (KCB.LE.NPIV_GLOBAL .AND.
     &          KCB+NELIM-1.GT.NPIV_GLOBAL) THEN
               CALL dgemm('T', 'N', NPIV_GLOBAL-KCB+1, NRHS_B,
     &              DIAGSIZ_DYN, MONE,
     &              DIAG(1+DIAGSIZ_DYN*LDADIAG), DIAGSIZ_DYN, 
     &              WCB(PPIV), LD_WCBPIV,
     &              ONE, WCB(PCB), LD_CB) 
               CALL dgemm('T', 'N', KCB+NELIM-NPIV_GLOBAL-1,
     &              NRHS_B, DIAGSIZ_DYN, MONE,
     &              DIAG(1+DIAGSIZ_DYN*LDADIAG +
     &              (NPIV_GLOBAL-KCB+1)*DIAGSIZ_DYN),
     &              DIAGSIZ_DYN, 
     &              WCB(PPIV), LD_WCBPIV,
     &              ONE, WCB(PCB_INIT), LD_WCBCB) 
             ELSE
               CALL dgemm('T', 'N', NELIM, NRHS_B, DIAGSIZ_DYN, MONE,
     &              DIAG(1+DIAGSIZ_DYN*LDADIAG), DIAGSIZ_DYN, 
     &              WCB(PPIV), LD_WCBPIV,
     &              ONE, WCB(PCB), LD_CB) 
             ENDIF
           ELSE
             IF (KCB.LE.NPIV_GLOBAL .AND.
     &          KCB+NELIM-1.GT.NPIV_GLOBAL) THEN
               CALL dgemm('N', 'N', NPIV_GLOBAL-KCB+1,
     &              NRHS_B, DIAGSIZ_DYN, MONE,
     &              DIAG(1+DIAGSIZ_DYN), DIAGSIZ_STA, 
     &              WCB(PPIV), LD_WCBPIV,
     &              ONE, WCB(PCB), LD_CB) 
               CALL dgemm('N', 'N', KCB+NELIM-NPIV_GLOBAL-1,
     &              NRHS_B, DIAGSIZ_DYN, MONE,
     &              DIAG(1+DIAGSIZ_DYN+NPIV_GLOBAL-KCB+1), 
     &              DIAGSIZ_STA, 
     &              WCB(PPIV), LD_WCBPIV,
     &              ONE, WCB(PCB_INIT), LD_WCBCB) 
             ELSE
               CALL dgemm('N', 'N', NELIM, NRHS_B, DIAGSIZ_DYN, MONE,
     &              DIAG(1+DIAGSIZ_DYN), DIAGSIZ_STA, 
     &              WCB(PPIV), LD_WCBPIV,
     &              ONE, WCB(PCB), LD_CB) 
             ENDIF
           ENDIF
         ENDIF
        CALL DMUMPS_SOL_FWD_BLR_UPDATE (
     &        WCB, LWCB, 1, LD_WCBPIV, PPIV_INIT, 1,
     &        WCB, LWCB, LD_WCBCB, PCB_INIT,
     &        PPIV, 
     &        NRHS_B, NPIV_GLOBAL,
     &        BLR_PANEL, LAST_BLR, I,
     &        BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC,
     &        KEEP8, KEEP(34), 
     &        KEEP(448), KEEP(450), KEEP(451), 
     &        .FALSE.,   
     &        IFLAG, IERROR)
        IF (IFLAG.LT.0) RETURN
        CALL DMUMPS_SOL_LD_AND_RELOAD_PANEL (
     &  INODE, N, DIAGSIZ_DYN, LIELL, NELIM, NSLAVES,
     &  PPIV,
     &  IW, IPOS, LIW, 
     &  DIAG(1), int(size(DIAG),8), 1_8, 
     &  WCB, LWCB, LD_WCBPIV, 
     &  RHSINTR, LRHSINTR, NRHS, 
     &  POSINRHSINTR_FWD, JBDEB, JBFIN, 
     &  MTYPE, KEEP, OOCWRITE_COMPATIBLE_WITH_BLR,
     &  .TRUE.  
     &  )
        PPIV = PPIV +  int(DIAGSIZ_DYN,8)
        IPOS = IPOS + DIAGSIZ_DYN
      ENDDO
      RETURN
      END SUBROUTINE DMUMPS_SOL_FWD_LR_SU
      SUBROUTINE DMUMPS_SOL_SLAVE_LR_U
     &    (INODE, IWHDLR, NPIV_GLOBAL,
     &     WCB, LWCB, 
     &     LDX, LDY,
     &     PTRX_INIT, PTRY_INIT, 
     &     JBDEB, JBFIN, 
     &     MTYPE, KEEP, KEEP8, IFLAG, IERROR )
       INTEGER, INTENT(IN) :: INODE, IWHDLR, NPIV_GLOBAL
       INTEGER, INTENT(IN) :: MTYPE, KEEP(500)
       INTEGER(8), INTENT(IN) :: KEEP8(150)
       INTEGER(8), INTENT(IN) :: LWCB, PTRX_INIT, PTRY_INIT
       INTEGER, INTENT(IN) :: LDX, LDY, JBDEB, JBFIN
       DOUBLE PRECISION, INTENT(INOUT) :: WCB(LWCB)
       INTEGER, INTENT(INOUT) :: IFLAG, IERROR
       INTEGER :: I, NPARTSASS, NB_BLR , NRHS_B
       INTEGER(8) :: PTRX, PTRY
       TYPE(LRB_TYPE), POINTER, DIMENSION(:) :: BLR_PANEL
      DOUBLE PRECISION :: ONE, MONE, ZERO
      PARAMETER (ONE = 1.0D0, MONE=-1.0D0)
      PARAMETER (ZERO=0.0D0)
      NRHS_B = JBFIN-JBDEB+1
      IF (associated(BLR_ARRAY(IWHDLR)%PANELS_L))
     &   THEN
         NPARTSASS=size(BLR_ARRAY(IWHDLR)%PANELS_L)
         NB_BLR = size(BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC)
         NB_BLR = NB_BLR - 2
      ELSE
        WRITE(6,*) " Internal error 1 in DMUMPS_SOL_SLAVE_LR_U"
        CALL MUMPS_ABORT()
      ENDIF
      PTRX = PTRX_INIT
      PTRY = PTRY_INIT
      DO I = 1, NPARTSASS
        BLR_PANEL => BLR_ARRAY(IWHDLR)%PANELS_L(I)%LRB_PANEL
        IF (associated(BLR_PANEL)) THEN
          IF (MTYPE.EQ.1) THEN 
            CALL DMUMPS_SOL_FWD_BLR_UPDATE (
     &          WCB, LWCB, 1, LDX, -99999_8, 1,
     &          WCB, LWCB, LDY, PTRY,
     &          PTRX,
     &          NRHS_B, NPIV_GLOBAL,
     &          BLR_PANEL, NB_BLR, 0, 
     &          BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC(2:NB_BLR+2),
     &          KEEP8, KEEP(34), 
     &          KEEP(448), KEEP(450), KEEP(451), 
     &          .TRUE., IFLAG, IERROR )
          ELSE 
            CALL DMUMPS_SOL_BWD_BLR_UPDATE (
     &          WCB, LWCB, 1, LDY, -99999_8, 1,
     &          WCB, LWCB, LDX, PTRX,
     &          PTRY,
     &          NRHS_B, NPIV_GLOBAL,
     &          BLR_PANEL, NB_BLR, 0, 
     &          BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC(2:NB_BLR+2),
     &          KEEP8, KEEP(34), 
     &          KEEP(448), KEEP(450), KEEP(451),
     &          .TRUE.,
     &          IFLAG, IERROR )
          ENDIF
          IF (MTYPE .EQ. 1) THEN
            PTRX = PTRX + BLR_PANEL(1)%N
          ELSE
            PTRY = PTRY + BLR_PANEL(1)%N
          ENDIF
          IF (IFLAG.LT.0) RETURN
        ENDIF
      ENDDO  
      RETURN
      END SUBROUTINE DMUMPS_SOL_SLAVE_LR_U
      SUBROUTINE GEMM_Q_FWD(m, nrhs_b, k, npiv, 
     &          Q, TMP, ldT,
     &          arraypiv, ldpiv, arraycb, lcb, ldcb, 
     &          ibeg_block, iend_block, is_t2_slave, 
     &          poscb, pospiv, pospivcol, ibeg_tmp)
          implicit none
          integer, intent(in) :: m, nrhs_b, k, npiv
          DOUBLE PRECISION, dimension(:,:), intent(inout)     :: Q
          DOUBLE PRECISION, dimension(ldt, *), intent(inout)     :: TMP
          integer(8), intent(in) :: lcb
          integer, intent(in)    :: ldpiv
          DOUBLE PRECISION, intent(inout) :: arraypiv(ldpiv,*)
          DOUBLE PRECISION, intent(inout) :: arraycb(lcb)
          integer, intent(in) :: ldt, ldcb
          integer, intent(in) :: ibeg_block, iend_block
          logical, intent(in) :: is_t2_slave
          integer(8), intent(in) :: poscb, pospiv
          integer, intent(in)    :: pospivcol
          integer, intent(in) :: ibeg_tmp
          integer :: posblock
          DOUBLE PRECISION :: ONE, MONE, ZERO
          PARAMETER (ONE = 1.0D0, MONE=-1.0D0)
          PARAMETER (ZERO=0.0D0)
          IF (IS_T2_SLAVE) THEN
            POSBLOCK = POSCB+int(IBEG_BLOCK-1,8)
            CALL dgemm('N', 'N', M, NRHS_B, K, MONE,
     &        Q(1,1), M, TMP(ibeg_tmp,1), LDT,
     &        ONE, ARRAYCB(POSBLOCK), LDCB) 
          ELSEIF (IBEG_BLOCK.LE.NPIV.AND.
     &            IEND_BLOCK.GT.NPIV) THEN
            POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
            CALL dgemm('N', 'N', NPIV-IBEG_BLOCK+1,NRHS_B, K,
     &        MONE, Q(1,1), M, TMP(ibeg_tmp,1), LDT,
     &        ONE, ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV) 
            CALL dgemm('N', 'N', IBEG_BLOCK+M-NPIV-1, 
     &        NRHS_B, K, 
     &        MONE, Q(NPIV-IBEG_BLOCK+2,1), M, 
     &        TMP(ibeg_tmp,1), LDT, ONE, ARRAYCB(POSCB), LDCB) 
          ELSEIF (IBEG_BLOCK.LE.NPIV) THEN
            POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
            CALL dgemm('N', 'N', M, NRHS_B, K, MONE,
     &        Q(1,1), M, TMP(ibeg_tmp,1), LDT,
     &        ONE, ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV) 
          ELSE
            POSBLOCK = POSCB+int(IBEG_BLOCK-1-NPIV,8)
            CALL dgemm('N', 'N', M, NRHS_B, K, MONE,
     &        Q(1,1), M, TMP(ibeg_tmp,1), LDT,
     &        ONE, ARRAYCB(POSBLOCK), LDCB) 
          ENDIF
      RETURN
      END SUBROUTINE GEMM_Q_FWD
      SUBROUTINE GEMM_Q_BWD(m, nrhs_b, k, npiv, 
     &          Q, TMP, ldT,
     &          arraypiv, lpiv, ldpiv, arraycb, lcb, ldcb, 
     &          ibeg_block, iend_block, is_t2_slave, 
     &          poscb, pospiv, pospivcol, ibeg_tmp)
          implicit none
          integer, intent(in) :: m, nrhs_b, k, npiv
          DOUBLE PRECISION, dimension(:, :), intent(inout)     :: Q
          DOUBLE PRECISION, dimension(ldt, *), intent(inout)     :: TMP
          integer(8), intent(in) :: lcb, lpiv
          DOUBLE PRECISION, intent(inout) :: arraypiv(lpiv,*)
          DOUBLE PRECISION, intent(inout) :: arraycb(lcb)
          integer, intent(in) :: ldt, ldcb, ldpiv
          integer, intent(in) :: ibeg_block, iend_block
          logical, intent(in) :: is_t2_slave
          integer(8), intent(in) :: poscb, pospiv
          integer, intent(in)    :: pospivcol
          integer, intent(in) :: ibeg_tmp
          integer(8) :: posblock
          DOUBLE PRECISION :: ONE, MONE, ZERO
          PARAMETER (ONE = 1.0D0, MONE=-1.0D0)
          PARAMETER (ZERO=0.0D0)
          IF (IS_T2_SLAVE) THEN
            POSBLOCK = POSCB +int(IBEG_BLOCK-1,8)
            CALL dgemm('T', 'N', K, NRHS_B, M, ONE,
     &        Q(1,1), M,  
     &        ARRAYCB(POSBLOCK), LDCB, ZERO, 
     &        TMP(ibeg_tmp,1), ldt)
          ELSEIF (IBEG_BLOCK.LE.NPIV.AND.
     &            IEND_BLOCK.GT.NPIV) THEN
            POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
            CALL dgemm('T', 'N', K, NRHS_B, 
     &        NPIV-IBEG_BLOCK+1,
     &        ONE, Q(1,1), M,  
     &        ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV,
     &        ZERO, TMP(ibeg_tmp, 1), ldt)
            CALL dgemm('T', 'N', 
     &        K, NRHS_B, IBEG_BLOCK+M-NPIV-1,
     &        ONE, Q(NPIV-IBEG_BLOCK+2,1), M,  
     &        ARRAYCB(POSCB), LDCB, 
     &        ONE,
     &        TMP(ibeg_tmp,1), ldt)
          ELSEIF (IBEG_BLOCK.LE.NPIV) THEN
            POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
            CALL dgemm('T', 'N', K, NRHS_B, M, ONE,
     &        Q(1,1), M,  
     &        ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV,
     &        ZERO, TMP(ibeg_tmp, 1), ldt)
          ELSE
            POSBLOCK = POSCB+int(IBEG_BLOCK-1-NPIV,8)
            CALL dgemm('T', 'N', K, NRHS_B, M, ONE,
     &        Q(1,1), M,  
     &        ARRAYCB(POSBLOCK), LDCB, ZERO, 
     &        TMP(ibeg_tmp, 1), ldt)
          ENDIF
      RETURN
      END SUBROUTINE GEMM_Q_BWD
      SUBROUTINE DMUMPS_SOL_FWD_BLR_UPDATE (
     &        ARRAYPIV, LPIV, LPIVCOL, LDPIV, POSPIV, POSPIVCOL,
     &        ARRAYCB, LCB, LDCB, POSCB, 
     &        POSDIAG,
     &        NRHS_B, NPIV,
     &        BLR_PANEL, LAST_BLR,
     &        CURRENT_BLR, BEGS_BLR_STATIC,
     &        KEEP8, K34, K448, K450, K451, IS_T2_SLAVE, IFLAG, IERROR )
!$    USE OMP_LIB
      INTEGER(8), INTENT(IN) :: LPIV, LCB, POSPIV, POSCB, POSDIAG
      INTEGER, INTENT(IN)    :: LPIVCOL, POSPIVCOL
      DOUBLE PRECISION, INTENT(INOUT) :: ARRAYPIV(LPIV,LPIVCOL)
      DOUBLE PRECISION, INTENT(INOUT) :: ARRAYCB(LCB)
      INTEGER, INTENT(IN)    :: LAST_BLR, NRHS_B,  LDPIV, LDCB,
     &                          CURRENT_BLR, NPIV, K34, K448, K450, K451
      TYPE(LRB_TYPE), TARGET,INTENT(IN) ::
     &                           BLR_PANEL(:)
      INTEGER :: BEGS_BLR_STATIC(:)
      LOGICAL, INTENT(IN) :: IS_T2_SLAVE
      INTEGER(8), INTENT(IN) :: KEEP8(150)
      INTEGER, INTENT(INOUT) :: IFLAG, IERROR
      INTEGER :: I, K, M, N, IBEG_BLOCK, IEND_BLOCK
      INTEGER :: KMAX
      INTEGER :: MMAX
      INTEGER(8) :: POSBLOCK
      INTEGER :: allocok
      TYPE(LRB_TYPE), POINTER :: LRB
      DOUBLE PRECISION, ALLOCATABLE,DIMENSION(:) :: TEMP_BLOCK
      DOUBLE PRECISION :: ONE, MONE, ZERO
      PARAMETER (ONE = 1.0D0, MONE=-1.0D0)
      PARAMETER (ZERO=0.0D0)
      KMAX = -1
      MMAX = -1
      DO I = CURRENT_BLR+1, LAST_BLR
        KMAX = max(KMAX, BLR_PANEL(I-CURRENT_BLR)%K)
        MMAX = max(MMAX, BLR_PANEL(I-CURRENT_BLR)%M)
      ENDDO
#if ! defined(BLR_NOOPENMP)
!$OMP PARALLEL PRIVATE(TEMP_BLOCK, allocok
!$OMP& )
#endif 
      IF (KMAX.GT.0) THEN
        allocate(TEMP_BLOCK(KMAX*NRHS_B), stat=allocok )
        IF (allocok .GT. 0) THEN
          IFLAG  = -13
          IERROR = NRHS_B * KMAX
          write(*,*) 'Allocation problem in BLR routine 
     &    DMUMPS_SOL_FWD_BLR_UPDATE for TEMP_BLOCK: ',
     &    'not enough memory? memory requested = ', IERROR
        ENDIF
      ENDIF
#if ! defined(BLR_NOOPENMP)
!$OMP BARRIER
#endif 
#if ! defined(BLR_NOOPENMP)
!$OMP DO SCHEDULE(DYNAMIC,1)
!$OMP& PRIVATE(IBEG_BLOCK, IEND_BLOCK, LRB, K, M, N,
!$OMP&         POSBLOCK)
#endif
        DO I = CURRENT_BLR+1, LAST_BLR
          IF (IFLAG.LT.0) CYCLE
          IBEG_BLOCK = BEGS_BLR_STATIC(I)
          IEND_BLOCK = BEGS_BLR_STATIC(I+1)-1
          IF (IBEG_BLOCK .EQ. IEND_BLOCK + 1) CYCLE
          LRB => BLR_PANEL(I-CURRENT_BLR)
          K = LRB%K 
          M = LRB%M
          N = LRB%N 
          IF (LRB%ISLR) THEN
            IF (K.GT.0) THEN
                CALL dgemm('N', 'N', K, NRHS_B, N, ONE,
     &                LRB%R(1,1), K, ARRAYPIV(POSDIAG,POSPIVCOL), LDPIV,
     &                ZERO, TEMP_BLOCK(1), K) 
                IF (IS_T2_SLAVE) THEN
                  POSBLOCK = POSCB+int(IBEG_BLOCK-1,8)
                  CALL dgemm('N', 'N', M, NRHS_B, K, MONE,
     &              LRB%Q(1,1), M, TEMP_BLOCK(1), K,
     &              ONE, ARRAYCB(POSBLOCK), LDCB) 
                ELSEIF (IBEG_BLOCK.LE.NPIV.AND.IEND_BLOCK.GT.NPIV) THEN
                  POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
                  CALL dgemm('N', 'N', NPIV-IBEG_BLOCK+1, NRHS_B, K,
     &              MONE, LRB%Q(1,1), M, TEMP_BLOCK(1), K,
     &              ONE, ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV) 
                  CALL dgemm('N', 'N', IBEG_BLOCK+M-NPIV-1, NRHS_B, K, 
     &              MONE, LRB%Q(NPIV-IBEG_BLOCK+2,1), M, TEMP_BLOCK(1),
     &              K, ONE, ARRAYCB(POSCB), LDCB) 
                ELSEIF (IBEG_BLOCK.LE.NPIV) THEN
                  POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
                  CALL dgemm('N', 'N', M, NRHS_B, K, MONE,
     &              LRB%Q(1,1), M, TEMP_BLOCK(1), K,
     &              ONE, ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV) 
                ELSE
                  POSBLOCK = POSCB+int(IBEG_BLOCK-1-NPIV,8)
                  CALL dgemm('N', 'N', M, NRHS_B, K, MONE,
     &              LRB%Q(1,1), M, TEMP_BLOCK(1), K,
     &              ONE, ARRAYCB(POSBLOCK), LDCB) 
                ENDIF
            ENDIF
          ELSE 
            IF (IS_T2_SLAVE) THEN
              POSBLOCK = POSCB + int(IBEG_BLOCK-1,8)
              CALL dgemm('N', 'N', M, NRHS_B, N, MONE,
     &              LRB%Q(1,1), M,  ARRAYPIV(POSDIAG,POSPIVCOL), LDPIV,
     &              ONE, ARRAYCB(POSBLOCK), LDCB) 
            ELSEIF (IBEG_BLOCK.LE.NPIV.AND.IEND_BLOCK.GT.NPIV) THEN
              POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
              CALL dgemm('N', 'N', NPIV-IBEG_BLOCK+1, NRHS_B, N, MONE,
     &              LRB%Q(1,1), M,  ARRAYPIV(POSDIAG,POSPIVCOL), LDPIV,
     &              ONE, ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV) 
              CALL dgemm('N', 'N', IBEG_BLOCK+M-NPIV-1, NRHS_B, N, MONE,
     &              LRB%Q(NPIV-IBEG_BLOCK+2,1), M,
     &              ARRAYPIV(POSDIAG,POSPIVCOL),
     &              LDPIV, ONE, ARRAYCB(POSCB), LDCB) 
            ELSEIF (IBEG_BLOCK.LE.NPIV) THEN
              POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
              CALL dgemm('N', 'N', M, NRHS_B, N, MONE,
     &              LRB%Q(1,1), M,  ARRAYPIV(POSDIAG,POSPIVCOL), LDPIV,
     &              ONE, ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV)
            ELSE
              POSBLOCK = POSCB + int(IBEG_BLOCK-1-NPIV,8)
              CALL dgemm('N', 'N', M, NRHS_B, N, MONE,
     &              LRB%Q(1,1), M,  ARRAYPIV(POSDIAG,POSPIVCOL), LDPIV,
     &              ONE, ARRAYCB(POSBLOCK), LDCB) 
            ENDIF
          ENDIF
        ENDDO
#if ! defined(BLR_NOOPENMP)
!$OMP END DO
#endif
        IF (KMAX.GT.0) THEN
          IF (allocated(TEMP_BLOCK)) deallocate(TEMP_BLOCK)
        ENDIF
#if ! defined(BLR_NOOPENMP)
!$OMP END PARALLEL         
#endif 
        RETURN
      END SUBROUTINE DMUMPS_SOL_FWD_BLR_UPDATE
      SUBROUTINE DMUMPS_SOL_BWD_LR_SU 
     &   ( INODE, IWHDLR, NPIV_GLOBAL, NSLAVES,
     &     LIELL, WCB, LWCB, NRHS_B, PTWCB,
     &     RHSINTR, LRHSINTR, NRHS,
     &     IPOSINRHSINTR, JBDEB, 
     &     MTYPE, KEEP, KEEP8,
     &     IFLAG, IERROR )
       INTEGER, INTENT(IN) :: INODE, IWHDLR, NPIV_GLOBAL, NSLAVES
       INTEGER, INTENT(IN) :: MTYPE, LIELL, KEEP(500)
       INTEGER(8), INTENT(IN) :: KEEP8(150)
       INTEGER, INTENT(IN) :: IPOSINRHSINTR, JBDEB, LRHSINTR, NRHS
       INTEGER(8), INTENT(IN) :: LWCB, PTWCB
       INTEGER, INTENT(IN) ::  NRHS_B
       INTEGER, INTENT(INOUT) :: IFLAG, IERROR
       DOUBLE PRECISION, INTENT(INOUT) :: WCB(LWCB)
       DOUBLE PRECISION RHSINTR(LRHSINTR,NRHS)
       INTEGER :: I, NPARTSASS, NB_BLR, LAST_BLR, 
     &            NELIM_PANEL, LD_WCB,
     &            DIAGSIZ_DYN, DIAGSIZ_STA, LDADIAG,
     &            IEND_BLR, IBEG_BLR
       INTEGER(8) :: PWCB
       INTEGER    :: IPIV_PANEL
       DOUBLE PRECISION, POINTER, DIMENSION(:) :: DIAG
       TYPE(LRB_TYPE), POINTER, DIMENSION(:) :: BLR_PANEL
       DOUBLE PRECISION :: ONE, MONE, ZERO
       PARAMETER (ONE = 1.0D0, MONE=-1.0D0)
       PARAMETER (ZERO=0.0D0)
       IF ((MTYPE.EQ.1).AND.(KEEP(50).EQ.0)) THEN
         IF (associated(BLR_ARRAY(IWHDLR)%PANELS_U))
     &      THEN
            NPARTSASS=size(BLR_ARRAY(IWHDLR)%PANELS_U)
            NB_BLR = size(BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC) -1
         ENDIF
       ELSE
         IF (associated(BLR_ARRAY(IWHDLR)%PANELS_L))
     &      THEN
            NPARTSASS=size(BLR_ARRAY(IWHDLR)%PANELS_L)
            NB_BLR = size(BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC) -1
         ELSE
           WRITE(6,*) " Internal error in DMUMPS_SOL_FWD_SU_MASTER"
         ENDIF
       ENDIF
      PWCB        = PTWCB + int(NPIV_GLOBAL,8)
      LD_WCB      = LIELL
      IF (KEEP(50).EQ.0 .AND. NSLAVES.GT.0 .AND. MTYPE.NE.1) THEN
        LAST_BLR = NPARTSASS
      ELSE
        LAST_BLR = NB_BLR
      ENDIF
      DO I=NPARTSASS,1,-1
        IBEG_BLR = BLR_ARRAY(IWHDLR)%BEGS_BLR_DYNAMIC(I)
        IEND_BLR =  BLR_ARRAY(IWHDLR)%BEGS_BLR_DYNAMIC(I+1) -1
        DIAGSIZ_DYN = BLR_ARRAY(IWHDLR)%BEGS_BLR_DYNAMIC(I+1) -
     &           IBEG_BLR
        DIAGSIZ_STA = BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC(I+1)  - 
     &           IBEG_BLR 
        IF (KEEP(50).NE.0) THEN
         LDADIAG = DIAGSIZ_DYN
        ELSE
         LDADIAG = DIAGSIZ_STA
        ENDIF 
        IF (DIAGSIZ_DYN.EQ.0) GOTO 1000
        NELIM_PANEL = DIAGSIZ_STA - DIAGSIZ_DYN
         IPIV_PANEL = IPOSINRHSINTR + IBEG_BLR -1
         IF ( MTYPE .EQ. 1 .AND. KEEP(50).EQ.0) THEN
          BLR_PANEL => BLR_ARRAY(IWHDLR)%PANELS_U(I)%LRB_PANEL
         ELSE
          BLR_PANEL => BLR_ARRAY(IWHDLR)%PANELS_L(I)%LRB_PANEL
         END IF
        CALL DMUMPS_SOL_BWD_BLR_UPDATE ( 
     &        RHSINTR, int(LRHSINTR,8), NRHS, LRHSINTR, 
     &        int(IPOSINRHSINTR,8), JBDEB,
     &        WCB, LWCB, LD_WCB, PWCB, 
     &        int(IPIV_PANEL,8), 
     &        NRHS_B, NPIV_GLOBAL,
     &        BLR_PANEL, LAST_BLR,
     &        I, BLR_ARRAY(IWHDLR)%BEGS_BLR_STATIC,
     &        KEEP8, KEEP(34), 
     &        KEEP(448), KEEP(450), KEEP(451),
     &        .FALSE., 
     &        IFLAG, IERROR)
        IF (IFLAG.LT.0) RETURN
         DIAG => BLR_ARRAY(IWHDLR)%DIAG_BLOCKS(I)%DIAG_BLOCK
         IF (NELIM_PANEL.GT.0) THEN
           IF (MTYPE.EQ.1.AND.KEEP(50).EQ.0) THEN
             IF (IEND_BLR.EQ.NPIV_GLOBAL) THEN
               CALL dgemm('T', 'N', DIAGSIZ_DYN, NRHS_B, NELIM_PANEL, 
     &              MONE, DIAG(1+DIAGSIZ_DYN), DIAGSIZ_STA,  WCB(PWCB),
     &              LD_WCB, ONE , RHSINTR(IPIV_PANEL,JBDEB),LRHSINTR)
             ELSE
               IF (IEND_BLR+1.LE.NPIV_GLOBAL .AND.
     &            IEND_BLR+NELIM_PANEL.GT.NPIV_GLOBAL) THEN
                 CALL dgemm('T', 'N', DIAGSIZ_DYN, NRHS_B,
     &              NPIV_GLOBAL-IEND_BLR, 
     &              MONE, DIAG(1+DIAGSIZ_DYN), DIAGSIZ_STA, 
     &              RHSINTR(IPIV_PANEL+DIAGSIZ_DYN,JBDEB), LRHSINTR,
     &              ONE, RHSINTR(IPIV_PANEL,JBDEB), LRHSINTR)
                 CALL dgemm('T', 'N', DIAGSIZ_DYN, NRHS_B,
     &              IEND_BLR+NELIM_PANEL-NPIV_GLOBAL, 
     &              MONE, DIAG(1+DIAGSIZ_DYN+NPIV_GLOBAL-IEND_BLR),
     &              DIAGSIZ_STA, 
     &              WCB(PWCB), LD_WCB,
     &              ONE, RHSINTR(IPIV_PANEL,JBDEB), LRHSINTR)
               ELSE
                 CALL dgemm('T', 'N', DIAGSIZ_DYN, NRHS_B, NELIM_PANEL, 
     &              MONE, DIAG(1+DIAGSIZ_DYN), DIAGSIZ_STA, 
     &              RHSINTR(IPIV_PANEL+DIAGSIZ_DYN,JBDEB), LRHSINTR,
     &              ONE, RHSINTR(IPIV_PANEL,JBDEB), LRHSINTR)
               ENDIF
             ENDIF
           ELSE
             IF (IEND_BLR.EQ.NPIV_GLOBAL) THEN
               CALL dgemm('N', 'N', DIAGSIZ_DYN, NRHS_B, NELIM_PANEL, 
     &              MONE, DIAG(1+DIAGSIZ_DYN*LDADIAG), DIAGSIZ_DYN, 
     &              WCB(PWCB), LD_WCB, ONE, 
     &              RHSINTR(IPIV_PANEL,JBDEB), LRHSINTR)
             ELSE
               IF (IEND_BLR+1.LE.NPIV_GLOBAL .AND.
     &            IEND_BLR+NELIM_PANEL.GT.NPIV_GLOBAL) THEN
                 CALL dgemm('N', 'N', DIAGSIZ_DYN, NRHS_B,
     &              NPIV_GLOBAL-IEND_BLR, 
     &              MONE, DIAG(1+DIAGSIZ_DYN*LDADIAG), DIAGSIZ_DYN, 
     &              RHSINTR(IPIV_PANEL+DIAGSIZ_DYN,JBDEB), LRHSINTR,
     &              ONE, RHSINTR(IPIV_PANEL,JBDEB), LRHSINTR)
                 CALL dgemm('N', 'N', DIAGSIZ_DYN, NRHS_B,
     &              IEND_BLR+NELIM_PANEL-NPIV_GLOBAL, 
     &              MONE, DIAG(1+DIAGSIZ_DYN*LDADIAG +
     &                  (NPIV_GLOBAL-IEND_BLR)*DIAGSIZ_DYN),
     &              DIAGSIZ_DYN, 
     &              WCB(PWCB), LD_WCB,
     &              ONE, RHSINTR(IPIV_PANEL,JBDEB), LRHSINTR)
               ELSE
                 CALL dgemm('N', 'N', DIAGSIZ_DYN, NRHS_B, NELIM_PANEL, 
     &              MONE, DIAG(1+DIAGSIZ_DYN*LDADIAG), DIAGSIZ_DYN, 
     &              RHSINTR(IPIV_PANEL+DIAGSIZ_DYN,JBDEB), LRHSINTR,
     &              ONE, RHSINTR(IPIV_PANEL,JBDEB), LRHSINTR)
               ENDIF
             ENDIF
           ENDIF
         ENDIF
         IF (IFLAG.LT.0) RETURN
         CALL DMUMPS_SOLVE_BWD_LR_TRSOLVE  (
     &         DIAG(1), size(DIAG), DIAGSIZ_DYN, NELIM_PANEL, LIELL,
     &         NRHS_B, WCB, LWCB, 
     &         RHSINTR, LRHSINTR, NRHS,
     &         IPIV_PANEL, JBDEB, 
     &         MTYPE, KEEP )
 1000 CONTINUE         
      ENDDO  
      RETURN
      END SUBROUTINE DMUMPS_SOL_BWD_LR_SU
      SUBROUTINE DMUMPS_SOL_BWD_BLR_UPDATE (
     &        ARRAYPIV, LPIV, LPIVCOL, LDPIV, POSPIV, POSPIVCOL,
     &        ARRAYCB, LCB, LDCB, POSCB, 
     &        POSDIAG,
     &        NRHS_B, NPIV,
     &        BLR_PANEL, LAST_BLR, CURRENT_BLR,
     &        BEGS_BLR_STATIC,
     &        KEEP8, K34, K448, K450, K451, IS_T2_SLAVE,
     &        IFLAG, IERROR)
!$    USE OMP_LIB
      INTEGER(8), INTENT(IN) :: LPIV, LCB, POSPIV, POSCB, POSDIAG
      INTEGER,INTENT(IN)     :: LPIVCOL, POSPIVCOL
      DOUBLE PRECISION, INTENT(INOUT) :: ARRAYPIV(LPIV,LPIVCOL)
      DOUBLE PRECISION, INTENT(INOUT) :: ARRAYCB(LCB)
      INTEGER, INTENT(IN)    :: LAST_BLR, NRHS_B,  LDPIV, LDCB,
     &                          CURRENT_BLR, NPIV, K34, K448, K450, K451
      TYPE(LRB_TYPE), TARGET,INTENT(IN) ::
     &                          BLR_PANEL(:)
      INTEGER(8), INTENT(IN) :: KEEP8(150)
      LOGICAL, INTENT(IN)    :: IS_T2_SLAVE
      INTEGER :: BEGS_BLR_STATIC(:)
      INTEGER, INTENT(INOUT) :: IFLAG, IERROR
      INTEGER :: I, K, M, N, IBEG_BLOCK, IEND_BLOCK
      INTEGER :: KMAX
      INTEGER(8) :: POSBLOCK
      TYPE(LRB_TYPE), POINTER :: LRB
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: TEMP_BLOCK
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: DEST_ARRAY
      INTEGER :: allocok
      DOUBLE PRECISION :: ONE, MONE, ZERO
      PARAMETER (ONE = 1.0D0, MONE=-1.0D0)
      PARAMETER (ZERO=0.0D0)
      KMAX = -1
      DO I = CURRENT_BLR+1, LAST_BLR
        KMAX = max(KMAX, BLR_PANEL(I-CURRENT_BLR)%K)
      ENDDO
      IF (CURRENT_BLR.LT.LAST_BLR) THEN 
        N = BLR_PANEL(1)%N
      ELSE
        RETURN
      ENDIF
      allocate(DEST_ARRAY(N*NRHS_B),stat=allocok)
      IF (allocok .GT. 0) THEN
        IFLAG = -13
        IERROR = N * NRHS_B
        GOTO 100
      ENDIF
      DEST_ARRAY = ZERO
#if ! defined(BLR_NOOPENMP)
!$OMP PARALLEL PRIVATE(TEMP_BLOCK, allocok
!$OMP& )
#endif 
      IF (KMAX.GT.0) THEN
        allocate(TEMP_BLOCK(KMAX*NRHS_B), stat=allocok )
        IF (allocok .GT. 0) THEN
          IFLAG  = -13
          IERROR = NRHS_B * KMAX
          write(*,*) 'Allocation problem in BLR routine 
     &    DMUMPS_SOL_BWD_BLR_UPDATE: ',
     &    'not enough memory? memory requested = ', IERROR
        ENDIF
      ENDIF
#if ! defined(BLR_NOOPENMP)
!$OMP DO SCHEDULE(DYNAMIC,1)
!$OMP& PRIVATE(IBEG_BLOCK, IEND_BLOCK, LRB, K, M, POSBLOCK)
!$OMP& REDUCTION(+:DEST_ARRAY)
#endif
        DO I = CURRENT_BLR+1, LAST_BLR
          IF (IFLAG.LT.0) CYCLE
          IBEG_BLOCK = BEGS_BLR_STATIC(I)
          IEND_BLOCK = BEGS_BLR_STATIC(I+1)-1
          LRB => BLR_PANEL(I-CURRENT_BLR)
          K = LRB%K 
          M = LRB%M   
          IF (LRB%ISLR) THEN
            IF (K.GT.0) THEN
                IF (IS_T2_SLAVE) THEN
                  POSBLOCK = POSCB +int(IBEG_BLOCK-1,8)
                  CALL dgemm('T', 'N', K, NRHS_B, M, ONE,
     &              LRB%Q(1,1), M,  
     &              ARRAYCB(POSBLOCK), LDCB, ZERO, 
     &              TEMP_BLOCK(1), K)
                ELSEIF (IBEG_BLOCK.LE.NPIV.AND.IEND_BLOCK.GT.NPIV) THEN
                  POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
                  CALL dgemm('T', 'N', K, NRHS_B, NPIV-IBEG_BLOCK+1,
     &              ONE, LRB%Q(1,1), M,  
     &              ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV,
     &              ZERO, TEMP_BLOCK(1), K)
                  CALL dgemm('T', 'N', K, NRHS_B, IBEG_BLOCK+M-NPIV-1,
     &              ONE, LRB%Q(NPIV-IBEG_BLOCK+2,1), M,  
     &              ARRAYCB(POSCB), LDCB, 
     &              ONE,  
     &              TEMP_BLOCK(1), K)
                ELSEIF (IBEG_BLOCK.LE.NPIV) THEN
                  POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
                  CALL dgemm('T', 'N', K, NRHS_B, M, ONE,
     &              LRB%Q(1,1), M,  
     &              ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV,
     &              ZERO, TEMP_BLOCK(1), K)
                ELSE
                  POSBLOCK = POSCB+int(IBEG_BLOCK-1-NPIV,8)
                  CALL dgemm('T', 'N', K, NRHS_B, M, ONE,
     &              LRB%Q(1,1), M,  
     &              ARRAYCB(POSBLOCK), LDCB, ZERO, 
     &              TEMP_BLOCK(1), K)
                ENDIF
                CALL dgemm('T', 'N', N, NRHS_B, K, MONE,
     &            LRB%R(1,1), K, 
     &            TEMP_BLOCK(1), K, ONE,
     &            DEST_ARRAY(1), N)
            ENDIF 
          ELSE 
            IF (IS_T2_SLAVE) THEN
              POSBLOCK = POSCB+int(IBEG_BLOCK-1,8)
              CALL dgemm('T', 'N', N, NRHS_B, M, MONE,
     &              LRB%Q(1,1), M,  ARRAYCB(POSBLOCK), LDCB,
     &              ONE, DEST_ARRAY(1), N)
            ELSEIF (IBEG_BLOCK.LE.NPIV.AND.IEND_BLOCK.GT.NPIV) THEN
              POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
              CALL dgemm('T', 'N', N, NRHS_B, NPIV-IBEG_BLOCK+1, MONE,
     &              LRB%Q(1,1), M,  ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV,
     &              ONE, DEST_ARRAY(1), N)
              CALL dgemm('T', 'N', N, NRHS_B, IBEG_BLOCK+M-NPIV-1, MONE,
     &              LRB%Q(NPIV-IBEG_BLOCK+2,1), M,  ARRAYCB(POSCB), 
     &              LDCB, ONE, DEST_ARRAY(1), N)
            ELSEIF (IBEG_BLOCK.LE.NPIV) THEN
              POSBLOCK = POSPIV+int(IBEG_BLOCK-1,8)
              CALL dgemm('T', 'N', N, NRHS_B, M, MONE,
     &              LRB%Q(1,1), M,  ARRAYPIV(POSBLOCK,POSPIVCOL), LDPIV,
     &              ONE, DEST_ARRAY(1), N)
            ELSE
              POSBLOCK = POSCB+int(IBEG_BLOCK-1-NPIV,8)
              CALL dgemm('T', 'N', N, NRHS_B, M, MONE,
     &              LRB%Q(1,1), M,  ARRAYCB(POSBLOCK), LDCB,
     &              ONE, DEST_ARRAY(1), N)
            ENDIF
          ENDIF
        ENDDO
#if ! defined(BLR_NOOPENMP)
!$OMP END DO
#endif
        IF (KMAX.GT.0) THEN
          IF (allocated(TEMP_BLOCK)) deallocate(TEMP_BLOCK)
        ENDIF
#if ! defined(BLR_NOOPENMP)
!$OMP END PARALLEL
#endif 
        IF (IS_T2_SLAVE) THEN
          DO I=1,NRHS_B
            call daxpy(N, ONE, DEST_ARRAY((I-1)*N+1), 1,
     &                 ARRAYPIV(POSDIAG+(I-1)*LDPIV,POSPIVCOL), 1)
          ENDDO
        ELSE
          DO I=1,NRHS_B
            call daxpy(N, ONE, DEST_ARRAY((I-1)*N+1), 1,
     &                 ARRAYPIV(POSDIAG,POSPIVCOL+I-1), 1)
          ENDDO
        ENDIF
 100    CONTINUE
        IF (allocated(DEST_ARRAY)) DEALLOCATE(DEST_ARRAY)
        RETURN
      END SUBROUTINE DMUMPS_SOL_BWD_BLR_UPDATE
      END MODULE DMUMPS_SOL_LR
      SUBROUTINE DMUMPS_SOLVE_BWD_LR_TRSOLVE (
     &           DIAG, LDIAG, NPIV, NELIM, LIELL,
     &           NRHS_B, W, LWC, 
     &           RHSINTR, LRHSINTR, NRHS,
     &           PPIVINRHSINTR, JBDEB, 
     &           MTYPE, KEEP)
       INTEGER, INTENT(IN) :: MTYPE, LIELL, NPIV, NELIM, KEEP(500)
       INTEGER, INTENT(IN) :: NRHS_B, LDIAG
       INTEGER, INTENT(IN) :: PPIVINRHSINTR, JBDEB, LRHSINTR, NRHS
       INTEGER(8), INTENT(IN) ::  LWC
       DOUBLE PRECISION, INTENT(IN) :: DIAG(LDIAG)
       DOUBLE PRECISION, INTENT(INOUT) :: W(LWC)
       DOUBLE PRECISION RHSINTR(LRHSINTR,NRHS)
       INTEGER :: LDAJ
      DOUBLE PRECISION ONE
      PARAMETER (ONE = 1.0D0)
        IF ( MTYPE .eq. 1 ) THEN
         LDAJ =  NPIV + NELIM
         CALL dtrsm('L','L','T','N', NPIV, NRHS_B, ONE, DIAG(1),
     &              LDAJ, RHSINTR(PPIVINRHSINTR,JBDEB), 
     &              LRHSINTR)
        ELSE
         IF ( KEEP(50) .EQ. 0 ) THEN
           LDAJ=NPIV+NELIM
         ELSE
           LDAJ=NPIV
         ENDIF
         CALL dtrsm('L','U','N','U', NPIV, NRHS_B, ONE, DIAG(1),
     &            LDAJ, RHSINTR(PPIVINRHSINTR,JBDEB), LRHSINTR)
        END IF 
      RETURN
      END SUBROUTINE DMUMPS_SOLVE_BWD_LR_TRSOLVE
