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
      SUBROUTINE CMUMPS_ANA_ARROWHEADS_WRAPPER ( id, 
     &               GATHER_MATRIX_ALLOCATED   ) 
      USE CMUMPS_STRUC_DEF
      USE CMUMPS_ANA_AUX_M, ONLY:CMUMPS_ANA_N_DIST
      IMPLICIT NONE
      INCLUDE 'mpif.h'
      INCLUDE 'mumps_tags.h'
      INTEGER   :: MASTER
      PARAMETER( MASTER = 0 )
      TYPE(CMUMPS_STRUC), TARGET  :: id
      LOGICAL, INTENT(INOUT)      :: GATHER_MATRIX_ALLOCATED
      INTEGER, ALLOCATABLE, DIMENSION(:) :: NBINROW_TMP, NBINCOL_TMP
      INTEGER, DIMENSION(:), POINTER :: KEEP, ICNTL, INFO
      INTEGER(8), DIMENSION(:), POINTER :: KEEP8
      LOGICAL  :: I_AM_SLAVE
      INTEGER  :: allocok
      KEEP    => id%KEEP
      ICNTL   => id%ICNTL
      INFO    => id%INFO 
      KEEP8  => id%KEEP8
      I_AM_SLAVE = ( id%MYID .ne. MASTER  .OR.
     &     ( id%MYID .eq. MASTER .AND.
     &     id%KEEP(46) .eq. 1 ) )
      IF (KEEP(55) .EQ. 0) THEN
         ALLOCATE( NBINCOL_TMP( id%N ), NBINROW_TMP( id%N ),
     &             stat=allocok )
         IF (allocok.GT.0) THEN
           INFO(1)=-7
           CALL MUMPS_SET_IERROR(int(id%N,8)+int(id%N,8), INFO(2))
         ENDIF
         CALL MUMPS_PROPINFO( ICNTL(1), INFO(1), id%COMM, id%MYID )
         IF ( INFO(1).LT.0 ) GOTO 500
         CALL CMUMPS_ANA_N_DIST(id, NBINCOL_TMP, NBINROW_TMP)
         IF ( .NOT. I_AM_SLAVE ) THEN
           DEALLOCATE(NBINCOL_TMP)
           DEALLOCATE(NBINROW_TMP)
         ENDIF
         IF(id%MYID .EQ. MASTER) THEN
            IF ( (KEEP(244) .EQ. 1) .AND. (KEEP(54) .EQ. 3) ) THEN
                IF (GATHER_MATRIX_ALLOCATED) THEN
                    IF (associated(id%IRN)) THEN
                      DEALLOCATE(id%IRN)
                      NULLIFY(id%IRN)
                    ENDIF
                    IF (associated(id%JCN)) THEN
                      DEALLOCATE(id%JCN)
                      NULLIFY(id%JCN)
                    ENDIF
                    GATHER_MATRIX_ALLOCATED= .FALSE.
                ENDIF
            END IF
         END IF
      ENDIF
      IF ( I_AM_SLAVE ) THEN
         IF (KEEP(55) .EQ. 0) THEN
            CALL CMUMPS_ANA_DIST_ARROWHEADS( id%MYID,
     &           id%NSLAVES, id%N, id%PROCNODE_STEPS(1),
     &           id%STEP(1), id%FILS(1),
     &           id%ISTEP_TO_INIV2(1), id%I_AM_CAND(1),
     &           KEEP(1),KEEP8(1), ICNTL(1), id,
     &           NBINCOL_TMP, NBINROW_TMP )
            DEALLOCATE(NBINCOL_TMP)
            DEALLOCATE(NBINROW_TMP)
         ELSE
            CALL CMUMPS_ANA_DIST_ELEMENTS( id%MYID,
     &           id%NSLAVES, id%N, id%PROCNODE_STEPS(1),
     &           id%STEP(1),
     &           id%PTRAR(1),
     &           id%PTRAR(id%NELT+2 ),
     &           id%NELT, 
     &           id%FRTPTR(1), id%FRTELT(1),
     &           KEEP(1), KEEP8(1), ICNTL(1), id%KEEP(50) )
                 id%KEEP(193)=1;id%KEEP(194)=1
                 id%KEEP(195)=1; id%KEEP(196)=1
            ALLOCATE( id%PTR8ARR(1),
     &                id%NINCOLARR(1),
     &                id%NINROWARR(1),
     &                id%PTRDEBARR(1), stat=allocok)
            IF (allocok .GT. 0) THEN
              id%INFO(1)=-7
              id%INFO(2)=4
            ENDIF
         ENDIF
      ELSE
        KEEP8(26) = 0_8
        KEEP8(27) = 0_8
        ALLOCATE( id%PTR8ARR(1),
     &            id%NINCOLARR(1),
     &            id%NINROWARR(1),
     &            id%PTRDEBARR(1), stat=allocok)
        IF (allocok .GT. 0) THEN
          id%INFO(1)=-7
          id%INFO(2)=4
        ENDIF
      ENDIF
 500  CONTINUE
      IF (allocated(NBINROW_TMP)) DEALLOCATE(NBINROW_TMP)
      IF (allocated(NBINCOL_TMP)) DEALLOCATE(NBINCOL_TMP)
      RETURN
      END SUBROUTINE CMUMPS_ANA_ARROWHEADS_WRAPPER
      SUBROUTINE CMUMPS_ANA_DIST_ARROWHEADS( MYID, SLAVEF, N,
     &           PROCNODE, STEP, FILS, ISTEP_TO_INIV2,
     &           I_AM_CAND,
     &           KEEP, KEEP8, ICNTL, id, NINCOL_TMP, NINROW_TMP )
      USE CMUMPS_STRUC_DEF
      IMPLICIT NONE
      TYPE (CMUMPS_STRUC) :: id
      INTEGER MYID, N, SLAVEF
      INTEGER KEEP( 500 ), ICNTL( 60 )
      INTEGER(8) KEEP8(150)
      INTEGER PROCNODE( KEEP(28) ), STEP( N ), FILS( N )
      INTEGER, INTENT(INOUT) :: NINCOL_TMP( N )
      INTEGER, INTENT(INOUT) :: NINROW_TMP( N )
      INTEGER ISTEP_TO_INIV2(KEEP(71))
      LOGICAL I_AM_CAND(max(1,KEEP(56)))
      LOGICAL I_AM_SLAVE
      LOGICAL I_AM_CAND_LOC
      INTEGER MUMPS_TYPENODE, MUMPS_PROCNODE, MUMPS_TYPESPLIT
      EXTERNAL MUMPS_TYPENODE, MUMPS_PROCNODE, MUMPS_TYPESPLIT
      INTEGER ISTEP, I, J, NINCOL, NINROW, allocok
      INTEGER TYPE_PARALL, ITYPE, IRANK, INIV2, TYPESPLIT 
      LOGICAL T4_MASTER_CONCERNED, EARLYT3ROOTINS
      INTEGER :: NBARR_LOCAL
      INTEGER(8) :: IPTR
      EARLYT3ROOTINS = KEEP(200) .EQ. 0
     & .OR. (KEEP(200) .LT. 0 .AND. KEEP(400) .EQ. 0)
      TYPE_PARALL = KEEP(46)
      I_AM_SLAVE = (KEEP(46).EQ.1 .OR. MYID.NE.0)
      NBARR_LOCAL=0
      DO J = 1, N
      ISTEP = STEP( J )
      IF ( ISTEP .GT. 0 ) THEN
      I = J
      DO WHILE (I .GT. 0)
        ITYPE = MUMPS_TYPENODE( PROCNODE(ISTEP), KEEP(199) )
        IRANK = MUMPS_PROCNODE( PROCNODE(ISTEP), KEEP(199) )
        TYPESPLIT = MUMPS_TYPESPLIT ( PROCNODE(ISTEP), KEEP(199) )
        I_AM_CAND_LOC = .FALSE.
        T4_MASTER_CONCERNED = .FALSE.
        IF (ITYPE.EQ.2) THEN
          INIV2         = ISTEP_TO_INIV2(ISTEP)
          IF (I_AM_SLAVE)  THEN
           I_AM_CAND_LOC = I_AM_CAND(INIV2)
           IF ( (TYPESPLIT.EQ.5).OR.(TYPESPLIT.EQ.6)) THEN
            IF ( TYPE_PARALL .eq. 0 ) THEN
             T4_MASTER_CONCERNED = 
     &       (id%CANDIDATES (id%CANDIDATES(SLAVEF+1,INIV2)+1,INIV2)
     &         .EQ.MYID-1 )
            ELSE
              T4_MASTER_CONCERNED = 
     &        (id%CANDIDATES (id%CANDIDATES(SLAVEF+1,INIV2)+1,INIV2) 
     &         .EQ.MYID )
            ENDIF
           ENDIF
          ENDIF
        ENDIF
        IF ( TYPE_PARALL .eq. 0 ) THEN
          IRANK =IRANK + 1
        END IF
        IF (
     &      ( ITYPE .eq. 2 .and.
     &        IRANK .eq. MYID )
     & .or.
     &      ( ITYPE .eq. 1 .and.
     &        IRANK .eq. MYID )
     & .or.
     &      ( T4_MASTER_CONCERNED )
     &     )  THEN
          NINCOL = NINCOL_TMP(I) 
          NINROW = NINROW_TMP(I)
        ELSE IF ( ITYPE .EQ. 3 ) THEN
          IF ( EARLYT3ROOTINS ) THEN
            NINCOL = -1
            NINROW = -1
          ELSE
            NINCOL = NINCOL_TMP(I)
            NINROW = NINROW_TMP(I)
          ENDIF
        ELSE IF ( ITYPE .eq. 2  .AND. I_AM_CAND_LOC ) THEN
          NINCOL = NINCOL_TMP(I)
          NINROW = 0
        ELSE
          NINCOL = -1
          NINROW = -1
        ENDIF
        IF ( NINCOL .NE. -1 ) THEN
          NBARR_LOCAL = NBARR_LOCAL + 1
        ENDIF
        NINCOL_TMP(I)=NINCOL
        NINROW_TMP(I)=NINROW
        I=FILS(I)
      ENDDO
      ENDIF
      ENDDO
      KEEP(193) = max(1, NBARR_LOCAL)
      KEEP(194) = max(1, NBARR_LOCAL)
      KEEP(195) = max(1, NBARR_LOCAL)
      KEEP(196) = KEEP(28)
      ALLOCATE(id%PTR8ARR(KEEP(193)),
     &         id%NINCOLARR(KEEP(194)), id%NINROWARR(KEEP(195)),
     &         id%PTRDEBARR(KEEP(196)), stat=allocok)
      IF (allocok.GT.0) THEN
        id%INFO(1)=-7
        CALL MUMPS_SET_IERROR( int(KEEP(194),8)+int(KEEP(195),8)+
     &                         int(KEEP(196),8), id%INFO(2) )
        RETURN
      ENDIF
      IPTR = 1_8       
      NBARR_LOCAL = 0  
      DO J = 1, N
        ISTEP = STEP( J )
        IF ( ISTEP .GT. 0 ) THEN
          id%PTRDEBARR(ISTEP) = NBARR_LOCAL + 1
          I = J
          DO WHILE (I .GT. 0)
            NINCOL = NINCOL_TMP(I)
            NINROW = NINROW_TMP(I)
            IF ( NINCOL .NE. -1 ) THEN
              NBARR_LOCAL = NBARR_LOCAL + 1
              id%NINCOLARR( NBARR_LOCAL ) = NINCOL
              id%NINROWARR( NBARR_LOCAL ) = NINROW
              id%PTR8ARR  ( NBARR_LOCAL ) = IPTR
              IPTR = IPTR + int(NINCOL + NINROW + 1,8)
            ENDIF
            I=FILS(I)
          ENDDO
          IF ( NINCOL .EQ. -1 ) THEN
            id%PTRDEBARR( ISTEP ) = -99999
          ENDIF
        ENDIF
      ENDDO
      KEEP8(26) = IPTR - 1
      KEEP8(27) = IPTR - 1
      RETURN
      END SUBROUTINE CMUMPS_ANA_DIST_ARROWHEADS
      SUBROUTINE CMUMPS_FACTO_SEND_ARROWHEADS( N, NZ, ASPK, 
     &   IRN, ICN, PERM,
     &   LSCAL,COLSCA,ROWSCA,
     &   MYID, SLAVEF, PROCNODE_STEPS, NBRECORDS,
     &   COMM, root, roota, KEEP, KEEP8, FILS,
     &   INTARR, LINTARR, DBLARR, LDBLARR,
     &   PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR, FRERE_STEPS,
     &   STEP, A, LA, ISTEP_TO_INIV2, I_AM_CAND, CANDIDATES,
     &   ICNTL, INFO )
!$    USE OMP_LIB
      USE MUMPS_INTR_TYPES, ONLY : MUMPS_ROOT_STRUC
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_ROOT_STRUC
      IMPLICIT NONE
      INTEGER    :: N, COMM, NBRECORDS
      INTEGER(8), INTENT(IN) :: NZ
      INTEGER KEEP( 500 )
      INTEGER(8) KEEP8(150)
      COMPLEX ASPK(NZ)
      REAL COLSCA(*), ROWSCA(*)
      INTEGER IRN(NZ), ICN(NZ) 
      INTEGER PERM(N), PROCNODE_STEPS(KEEP(28))
      INTEGER FILS( N )
      INTEGER ISTEP_TO_INIV2(KEEP(71))
      LOGICAL I_AM_CAND(max(1,KEEP(56)))
      INTEGER SLAVEF, MYID
      INTEGER CANDIDATES(SLAVEF+1, max(1,KEEP(56)))
      LOGICAL LSCAL
      TYPE (MUMPS_ROOT_STRUC) :: root
      TYPE (CMUMPS_ROOT_STRUC) :: roota
      INTEGER INFO( 80 ), ICNTL(60)
      INTEGER(8), INTENT(IN)    :: LA
      INTEGER(8), INTENT(IN) :: PTR8ARR(KEEP(193))
      INTEGER, INTENT(IN) :: NINCOLARR(KEEP(194))
      INTEGER, INTENT(IN) :: NINROWARR(KEEP(195))
      INTEGER, INTENT(IN) :: PTRDEBARR(KEEP(196))
      INTEGER    :: FRERE_STEPS( KEEP(28) )
      INTEGER    :: STEP(N)
      INTEGER(8) :: LINTARR, LDBLARR
      INTEGER    :: INTARR( LINTARR )
      COMPLEX    :: DBLARR( LDBLARR )
      COMPLEX    :: A( LA )
      INTEGER MUMPS_TYPESPLIT
      EXTERNAL MUMPS_TYPESPLIT
      INTEGER LP
      LOGICAL LPOK
      COMPLEX VAL, VAL_SHR
      INTEGER IOLD,JOLD,ISEND,JSEND,DEST,I,IARR
      INTEGER ISEND_SHR, JSEND_SHR, DEST_SHR
      INTEGER IPOSROOT, JPOSROOT
      INTEGER IROW_GRID, JCOL_GRID
      INTEGER ISTEP
      INTEGER NBUFS
      INTEGER ARROW_ROOT, TAILLE
      INTEGER LOCAL_M, LOCAL_N
      INTEGER(8) :: PTR_ROOT
      INTEGER TYPE_NODE, MASTER_NODE
      LOGICAL I_AM_CAND_LOC, I_AM_SLAVE
      INTEGER JARR, ILOCROOT, JLOCROOT
      INTEGER allocok, INIV2, TYPESPLIT, T4MASTER
      INTEGER(8) ::  IS8, K
      INTEGER NCAND
      LOGICAL T4_MASTER_CONCERNED, EARLYT3ROOTINS
      COMPLEX ZERO
      PARAMETER( ZERO = (0.0E0,0.0E0) )
      INTEGER, ALLOCATABLE, DIMENSION(:,:) :: IW4
      INTEGER(8), ALLOCATABLE, DIMENSION(:) :: PTRAW
      INTEGER :: IARR1, IORG, J
      INTEGER, DIMENSION(:,:), ALLOCATABLE :: BUFI
      COMPLEX, DIMENSION(:,:), ALLOCATABLE :: BUFR
      LOGICAL :: DOIT, OMP_FLAG, OMP_FLAG_P
      INTEGER NOMP, NOMP_P, IOMP, P2
      LP = ICNTL(1)
      LPOK = ( LP .GT. 0 .AND. ICNTL(4) .GE. 1 )
      ARROW_ROOT = 0
      EARLYT3ROOTINS = KEEP(200) .EQ. 0
     & .OR. (KEEP(200) .LT. 0 .AND. KEEP(400) .EQ. 0)
      I_AM_SLAVE=(MYID.NE.0.OR.KEEP(46).EQ.1)
      IF ( KEEP(46) .eq. 0 ) THEN
        NBUFS = SLAVEF
      ELSE
        NBUFS = SLAVEF - 1
        ALLOCATE( IW4( N, 2 ), stat = allocok )
        IF ( allocok .GT. 0 ) THEN
          INFO(1) = -13
          CALL MUMPS_SET_IERROR( int(N,8)+int(N,8), INFO(2) )
          IF (LPOK ) WRITE (LP,*) MYID,
     &    ': Error allocating IW4 in CMUMPS_FACTO_SEND_ARROWHEADS'
          GOTO 100
        END IF
        ALLOCATE( PTRAW( N ), stat = allocok )
        IF ( allocok .GT. 0 ) THEN
          INFO(1) = -13
          INFO(2) = N
          IF (LPOK ) WRITE (LP,*) MYID,
     &    ': Error allocating PTRAW in CMUMPS_FACTO_SEND_ARROWHEADS'
          GOTO 100
        END IF
      ENDIF
      IF (NBUFS.GT.0) THEN
       ALLOCATE( BUFI(NBRECORDS*2+1,NBUFS),stat=allocok )
       IF ( allocok .GT. 0 ) THEN
         IF (LPOK ) WRITE (LP,*) MYID,
     &   ': Error allocating BUFI in CMUMPS_FACTO_SEND_ARROWHEADS'
         INFO(1)=-13
         CALL MUMPS_SET_IERROR( int(NBUFS,8) * int(NBRECORDS*2+1,8),
     &                          INFO(2))
         GOTO 100
       END IF
       ALLOCATE( BUFR( NBRECORDS, NBUFS ), stat=allocok )
       IF ( allocok .GT. 0 ) THEN
         INFO(1) =-13
         CALL MUMPS_SET_IERROR( int(NBUFS,8) * int(NBRECORDS*2+1,8),
     &                          INFO(2))
         IF (LPOK ) WRITE (LP,*) MYID,
     &   ': Error allocating BUFR in CMUMPS_FACTO_SEND_ARROWHEADS'
         GOTO 100
       END IF
        DO I = 1, NBUFS
          BUFI( 1, I ) = 0
        ENDDO
      ENDIF
 100  CONTINUE
      CALL MUMPS_PROPINFO( ICNTL, INFO, COMM, MYID )
      IF ( INFO( 1 ) .LT. 0 ) GOTO 500
      IF (KEEP(46) .NE. 0) THEN
#if defined(__ve__)
!NEC$ IVDEP
#endif
        DO J = 1, N
          ISTEP=STEP(J)
          IF (ISTEP .GT. 0) THEN
            IARR1 = PTRDEBARR( ISTEP )
            IF ( IARR1 .GT. 0 ) THEN
              I = J
              IORG = 0
              DO WHILE ( I .GT. 0 )
                IORG = IORG + 1
                IW4(I, 1)  = NINCOLARR( IARR1 + IORG - 1 )
                IW4(I, 2)  = NINROWARR( IARR1 + IORG - 1 ) +
     &                       NINCOLARR( IARR1 + IORG - 1 )
                IS8        = PTR8ARR( IARR1 + IORG - 1 )
                PTRAW( I ) = IS8
                INTARR( IS8 ) = I
                DBLARR( IS8 ) = ZERO
                I = FILS(I)
              ENDDO
            ENDIF
          ENDIF
        ENDDO
        IF ( KEEP(38) .NE. 0 .AND. EARLYT3ROOTINS ) THEN
          CALL CMUMPS_GET_ROOT_INFO(root, LOCAL_M, LOCAL_N,
     &                              PTR_ROOT, LA)
          CALL CMUMPS_SET_ROOT_TO_ZERO(root, roota, KEEP, A, LA)
        ELSE
          LOCAL_M = -19999; LOCAL_N = -29999; PTR_ROOT = -99999_8
        END IF
      END IF
      NOMP = 1
!$    NOMP=omp_get_max_threads()
      OMP_FLAG = KEEP(399).EQ.1 .AND. NOMP.GE.2 .AND. SLAVEF.EQ.1
     &           .AND. KEEP(46) .EQ. 1
!$OMP PARALLEL PRIVATE(K, I, DEST, I_AM_CAND_LOC,
!$OMP&          T4MASTER, T4_MASTER_CONCERNED,
!$OMP&          INIV2, NCAND, IROW_GRID, JCOL_GRID,
!$OMP&          ILOCROOT, JLOCROOT, IPOSROOT, JPOSROOT,
!$OMP&          TYPE_NODE, TYPESPLIT, MASTER_NODE,
!$OMP&          IS8, TAILLE, VAL,
!$OMP&          IARR, JARR, ISTEP, ISEND, JSEND,
!$OMP&          IOLD, JOLD, IOMP, DOIT, P2, NOMP_P, OMP_FLAG_P)
!$OMP& REDUCTION(+: ARROW_ROOT) IF (OMP_FLAG)
      IOMP=0
!$    IOMP=omp_get_thread_num()
      NOMP_P=1
!$    NOMP_P=omp_get_num_threads()
      OMP_FLAG_P = .FALSE.
!$    OMP_FLAG_P = OMP_FLAG .AND. NOMP_P .GT. 1
      IF (OMP_FLAG_P) THEN
        IF ( NOMP_P .GE. 16 ) THEN
          NOMP_P=16
          P2 = 4
        ELSE IF (NOMP_P.GE.8) THEN
          NOMP_P=8
          P2 = 3
        ELSE IF (NOMP_P.GE.4) THEN
          NOMP_P=4
          P2 = 2
        ELSE IF (NOMP_P.GE.2) THEN
          NOMP_P=2
          P2 = 1
        ENDIF
      ELSE
        NOMP_P = 1
        P2 = 0
      ENDIF
      IF ( IOMP .LT. NOMP_P ) THEN
       DO K=1, NZ
        IOLD = IRN(K)
        JOLD = ICN(K)
        IF ( (IOLD.GT.N).OR.(JOLD.GT.N).OR.(IOLD.LT.1)
     &                 .OR.(JOLD.LT.1) ) THEN
          CYCLE
        END IF
        IF (OMP_FLAG_P) THEN
          IF (IOLD.EQ.JOLD) THEN
            IARR = IOLD
          ELSE IF (PERM(IOLD).LT.PERM(JOLD)) THEN
             IARR = IOLD
          ELSE
             IARR = JOLD
          ENDIF
          DOIT = ( IOMP .EQ. ibits(IARR, P2-1, P2))
        ELSE
          DOIT = .TRUE.
        ENDIF
        IF (DOIT) THEN
          IF (IOLD.EQ.JOLD) THEN
            ISEND = IOLD
            JSEND = JOLD
            IARR  = IOLD
          ELSE IF (PERM(IOLD).LT.PERM(JOLD)) THEN
            IARR = IOLD
            IF ( KEEP(50) .NE. 0 ) THEN
              ISEND = -IOLD
            ELSE
              ISEND = IOLD
            ENDIF
            JSEND = JOLD
          ELSE
            IARR = JOLD
            ISEND = -JOLD
            JSEND = IOLD
          ENDIF
          ISTEP = abs( STEP(IARR) )
          CALL MUMPS_TYPEANDPROCNODE( TYPE_NODE, MASTER_NODE,
     &    PROCNODE_STEPS(ISTEP), KEEP(199) ) 
          I_AM_CAND_LOC          = .FALSE.
          T4_MASTER_CONCERNED    = .FALSE.
          T4MASTER               = -9999
          IF ( TYPE_NODE .EQ. 1 ) THEN
            IF ( KEEP(46) .eq. 0 ) THEN
              DEST = MASTER_NODE + 1
            ELSE
              DEST = MASTER_NODE
            END IF
          ELSE IF ( TYPE_NODE .EQ. 2 ) THEN
            IF ( ISEND .LT. 0  ) THEN
              DEST = -1
            ELSE
              IF ( KEEP( 46 ) .eq. 0 ) THEN
                DEST = MASTER_NODE + 1
              ELSE 
                DEST = MASTER_NODE
              END IF
            END IF
            INIV2         = ISTEP_TO_INIV2(ISTEP)
            IF (I_AM_SLAVE) I_AM_CAND_LOC = I_AM_CAND(INIV2)
            IF ( KEEP(79) .GT. 0) THEN
              TYPESPLIT  = MUMPS_TYPESPLIT( PROCNODE_STEPS(ISTEP),
     &                                      KEEP(199) )
              IF ( (TYPESPLIT.EQ.5).OR.(TYPESPLIT.EQ.6)) THEN
                T4_MASTER_CONCERNED = .TRUE.
                T4MASTER=CANDIDATES(CANDIDATES(SLAVEF+1,INIV2)+1,INIV2)
                IF ( KEEP(46) .eq. 0 ) THEN
                 T4MASTER=T4MASTER+1
                ENDIF
              ENDIF
            ENDIF
          ELSE 
            ARROW_ROOT = ARROW_ROOT + 1
            IF (EARLYT3ROOTINS) THEN
              IF ( ISEND .LT. 0 ) THEN
                IPOSROOT = root%RG2L(JSEND)
                JPOSROOT = root%RG2L(IARR)
              ELSE
                IPOSROOT = root%RG2L( IARR )
                JPOSROOT = root%RG2L( JSEND )
              END IF
              IROW_GRID = mod( ( IPOSROOT-1 )/root%MBLOCK, root%NPROW )
              JCOL_GRID = mod( ( JPOSROOT-1 )/root%NBLOCK, root%NPCOL )
              IF ( KEEP( 46 ) .eq. 0 ) THEN
                DEST = IROW_GRID * root%NPCOL + JCOL_GRID + 1
              ELSE
                DEST = IROW_GRID * root%NPCOL + JCOL_GRID
              END IF
            ELSE
              DEST = -2
            ENDIF
          END IF
          IF (LSCAL) THEN
            VAL = ASPK(K)*ROWSCA(IOLD)*COLSCA(JOLD)
          ELSE
            VAL = ASPK(K)
          ENDIF
          IF ( DEST .eq. 0
     &       .or. 
     &        ( DEST .eq. -1 .and. KEEP( 46 ) .eq. 1 .AND.
     &         ( I_AM_CAND_LOC .OR. MASTER_NODE .EQ. 0 ) )
     &       .or. 
     &        ( T4MASTER.EQ.0 )
     &       .or. 
     &        ( DEST .EQ. -2 .AND. KEEP( 46 ) .EQ. 1 )
     &       ) THEN
            IARR = ISEND  
            JARR = JSEND
            IF ( TYPE_NODE .eq. 3 .AND. EARLYT3ROOTINS ) THEN
              IF ( IROW_GRID .EQ. root%MYROW .AND.
     &           JCOL_GRID .EQ. root%MYCOL ) THEN
                ILOCROOT = root%MBLOCK * ( ( IPOSROOT - 1 ) /
     &                   ( root%MBLOCK * root%NPROW ) )
     &                 + mod( IPOSROOT - 1, root%MBLOCK ) + 1
                JLOCROOT = root%NBLOCK * ( ( JPOSROOT - 1 ) /
     &                   ( root%NBLOCK * root%NPCOL ) )
     &                 + mod( JPOSROOT - 1, root%NBLOCK ) + 1
               IF (KEEP(60)==0) THEN
                 A( PTR_ROOT
     &             + int(JLOCROOT - 1,8) * int(LOCAL_M,8) 
     &             + int(ILOCROOT - 1,8) )
     &           =  A( PTR_ROOT
     &             + int(JLOCROOT - 1,8) * int(LOCAL_M,8)
     &             + int(ILOCROOT - 1,8) )
     &           + VAL
               ELSE
                 roota%SCHUR_POINTER( int(JLOCROOT - 1,8)
     &                             * int(root%SCHUR_LLD,8)
     &                             + int(ILOCROOT,8) )
     &            = roota%SCHUR_POINTER( int(JLOCROOT - 1,8)
     &                             *    int(root%SCHUR_LLD,8)
     &                             +    int(ILOCROOT,8))
     &            + VAL
               ENDIF
              ELSE
                WRITE(*,*) MYID,':INTERNAL Error: root arrowhead '
                WRITE(*,*) MYID,':is not belonging to me. IARR,JARR='
     &          ,IARR,JARR
                CALL MUMPS_ABORT()
              END IF
            ELSE IF ( IARR .GE. 0 ) THEN
              IF ( IARR .eq. JARR ) THEN
                IS8 = PTRAW( IARR )
                DBLARR( IS8 ) = DBLARR( IS8 ) + VAL
              ELSE
                IS8         =  PTRAW(IARR) + IW4(IARR,2)
                IW4(IARR,2) = IW4(IARR,2) - 1
                INTARR(IS8) = JARR
                DBLARR(IS8) = VAL
              END IF
            ELSE
              IARR        = -IARR
              IS8         = PTRAW(IARR)+IW4(IARR,1)
              IW4(IARR,1) = IW4(IARR,1) - 1
              INTARR(IS8) = JARR
              DBLARR(IS8) = VAL
              IF ( IW4(IARR,1) .EQ. 0 .AND.
     &             STEP( IARR) > 0 ) THEN
                IF ( MASTER_NODE == MYID) THEN
                  TAILLE = NINCOLARR(PTRDEBARR(STEP( IARR ) ))
                  CALL CMUMPS_QUICK_SORT_ARROWHEADS( N, PERM,
     &               INTARR( PTRAW(IARR) + 1 ),
     &               DBLARR( PTRAW(IARR) + 1 ),
     &               TAILLE, 1, TAILLE )
                END IF
              END IF
            END IF
          END IF
          IF ( DEST.EQ. -1 ) THEN
            INIV2 = ISTEP_TO_INIV2(ISTEP)
            NCAND = CANDIDATES(SLAVEF+1,INIV2)
            IF (KEEP(79).GT.0) THEN
              DO I=1, SLAVEF
                DEST=CANDIDATES(I,INIV2)
                IF (KEEP(46).EQ.0.AND.(DEST.GE.0)) DEST=DEST+1
                IF (DEST.LT.0) EXIT 
                IF (I.EQ.NCAND+1) CYCLE
                IF (DEST.NE.0) THEN
                  ISEND_SHR=ISEND; JSEND_SHR=JSEND
                  VAL_SHR=VAL; DEST_SHR=DEST
                  CALL CMUMPS_ARROW_FILL_SEND_BUF()
                ENDIF
              ENDDO
            ELSE
              DO I=1, NCAND
                DEST=CANDIDATES(I,INIV2)
                IF (KEEP(46).EQ.0) DEST=DEST+1
                IF (DEST.NE.0) THEN
                  ISEND_SHR=ISEND; JSEND_SHR=JSEND
                  VAL_SHR=VAL; DEST_SHR=DEST
                  CALL CMUMPS_ARROW_FILL_SEND_BUF()
                ENDIF
              ENDDO
            ENDIF
            DEST = MASTER_NODE
            IF (KEEP(46).EQ.0) DEST=DEST+1
            IF ( DEST .NE. 0 ) THEN
              ISEND_SHR=ISEND; JSEND_SHR=JSEND
              VAL_SHR=VAL; DEST_SHR=DEST
              CALL CMUMPS_ARROW_FILL_SEND_BUF()
            ENDIF
            IF ((T4_MASTER_CONCERNED).AND.(T4MASTER.GT.0)) THEN 
              ISEND_SHR=ISEND; JSEND_SHR=JSEND
              VAL_SHR=VAL; DEST_SHR=T4MASTER
              CALL CMUMPS_ARROW_FILL_SEND_BUF()
            ENDIF
          ELSE IF ( DEST .GT. 0 ) THEN
            ISEND_SHR=ISEND; JSEND_SHR=JSEND
            VAL_SHR=VAL; DEST_SHR=DEST
            CALL CMUMPS_ARROW_FILL_SEND_BUF()
            IF ( T4MASTER.GT.0 ) THEN
              ISEND_SHR=ISEND; JSEND_SHR=JSEND
              VAL_SHR=VAL; DEST_SHR=T4MASTER
              CALL CMUMPS_ARROW_FILL_SEND_BUF()
            ENDIF
          ELSE IF ( T4MASTER.GT.0 ) THEN
            ISEND_SHR=ISEND; JSEND_SHR=JSEND
            VAL_SHR=VAL; DEST_SHR=T4MASTER
            CALL CMUMPS_ARROW_FILL_SEND_BUF()
          ELSE IF ( DEST .EQ. -2 ) THEN
            DO I = 0, SLAVEF-1
              DEST = I
              IF (KEEP(46) .EQ. 0) DEST = DEST + 1
              IF (DEST .NE. 0) THEN
                ISEND_SHR=ISEND; JSEND_SHR=JSEND
                VAL_SHR=VAL; DEST_SHR=DEST
                CALL CMUMPS_ARROW_FILL_SEND_BUF()
              ENDIF
            ENDDO
          ENDIF
        ENDIF 
       ENDDO
      ENDIF
!$OMP END PARALLEL
       KEEP(49) = ARROW_ROOT
       IF (NBUFS.GT.0) THEN
        CALL CMUMPS_ARROW_FINISH_SEND_BUF(
     &   BUFI, BUFR, NBRECORDS, NBUFS,
     &   LP, COMM, KEEP( 46 ) )
      ENDIF
  500 CONTINUE
      IF ( allocated(IW4   ) ) DEALLOCATE( IW4   )
      IF ( allocated(PTRAW ) ) DEALLOCATE( PTRAW )
      IF ( allocated(BUFI  ) ) DEALLOCATE( BUFI  )
      IF ( allocated(BUFR  ) ) DEALLOCATE( BUFR  )
      RETURN
      CONTAINS
      SUBROUTINE CMUMPS_ARROW_FILL_SEND_BUF()
      IMPLICIT NONE
      INCLUDE 'mpif.h'
      INCLUDE 'mumps_tags.h'
      INTEGER IERR
      INTEGER TAILLE_SENDI, TAILLE_SENDR, IREQ
         IF (BUFI(1,DEST_SHR)+1.GT.NBRECORDS) THEN
          TAILLE_SENDI = BUFI(1,DEST_SHR) * 2 + 1
          TAILLE_SENDR = BUFI(1,DEST_SHR)
          CALL MPI_SEND(BUFI(1,DEST_SHR),TAILLE_SENDI,
     &                   MPI_INTEGER,
     &                   DEST_SHR, ARROWHEAD, COMM, IERR )
          CALL MPI_SEND( BUFR(1,DEST_SHR), TAILLE_SENDR,
     &                   MPI_COMPLEX, DEST_SHR,
     &                   ARROWHEAD, COMM, IERR )
          BUFI(1,DEST_SHR) = 0
         ENDIF
         IREQ = BUFI(1,DEST_SHR) + 1
         BUFI(1,DEST_SHR) = IREQ
         BUFI( IREQ * 2, DEST_SHR )     = ISEND_SHR
         BUFI( IREQ * 2 + 1, DEST_SHR ) = JSEND_SHR
         BUFR( IREQ, DEST_SHR )         = VAL_SHR
      RETURN
      END SUBROUTINE CMUMPS_ARROW_FILL_SEND_BUF
      END SUBROUTINE CMUMPS_FACTO_SEND_ARROWHEADS
      SUBROUTINE CMUMPS_ARROW_FILL_SEND_BUF_ELT(
     &   ISEND_SHR, JSEND_SHR, VAL_SHR,
     &   DEST_SHR, BUFI, BUFR, NBRECORDS, NBUFS, LP, COMM )
      IMPLICIT NONE
      INTEGER, INTENT(in) :: ISEND_SHR, JSEND_SHR
      COMPLEX, INTENT(in) :: VAL_SHR
      INTEGER :: DEST_SHR, NBRECORDS, NBUFS, LP, COMM
      INTEGER :: BUFI( NBRECORDS*2+1, NBUFS )
      COMPLEX :: BUFR( NBRECORDS, NBUFS )
      INCLUDE 'mpif.h'
      INCLUDE 'mumps_tags.h'
      INTEGER IERR
      INTEGER TAILLE_SENDI, TAILLE_SENDR, IREQ
         IF (BUFI(1,DEST_SHR)+1.GT.NBRECORDS) THEN
          TAILLE_SENDI = BUFI(1,DEST_SHR) * 2 + 1
          TAILLE_SENDR = BUFI(1,DEST_SHR)
          CALL MPI_SEND(BUFI(1,DEST_SHR),TAILLE_SENDI,
     &                   MPI_INTEGER,
     &                   DEST_SHR, ARROWHEAD, COMM, IERR )
          CALL MPI_SEND( BUFR(1,DEST_SHR), TAILLE_SENDR,
     &                   MPI_COMPLEX, DEST_SHR,
     &                   ARROWHEAD, COMM, IERR )
          BUFI(1,DEST_SHR) = 0
         ENDIF
         IREQ = BUFI(1,DEST_SHR) + 1
         BUFI(1,DEST_SHR) = IREQ
         BUFI( IREQ * 2, DEST_SHR )     = ISEND_SHR
         BUFI( IREQ * 2 + 1, DEST_SHR ) = JSEND_SHR
         BUFR( IREQ, DEST_SHR )         = VAL_SHR
      RETURN
      END SUBROUTINE CMUMPS_ARROW_FILL_SEND_BUF_ELT
      SUBROUTINE CMUMPS_ARROW_FINISH_SEND_BUF(
     &   BUFI, BUFR, NBRECORDS, NBUFS, LP, COMM,
     &   TYPE_PARALL )
      IMPLICIT NONE
      INTEGER NBUFS, NBRECORDS, TYPE_PARALL
      INTEGER BUFI( NBRECORDS * 2 + 1, NBUFS )
      COMPLEX BUFR( NBRECORDS, NBUFS )
      INTEGER COMM
      INTEGER LP
      INTEGER ISLAVE, TAILLE_SENDI, TAILLE_SENDR, IERR
      INCLUDE 'mpif.h'
      INCLUDE 'mumps_tags.h'
        DO ISLAVE = 1,NBUFS 
          TAILLE_SENDI = BUFI(1,ISLAVE) * 2 + 1
          TAILLE_SENDR = BUFI(1,ISLAVE)
          BUFI(1,ISLAVE) = - BUFI(1,ISLAVE)
          CALL MPI_SEND(BUFI(1,ISLAVE),TAILLE_SENDI,
     &                   MPI_INTEGER,
     &                   ISLAVE, ARROWHEAD, COMM, IERR )
          IF ( TAILLE_SENDR .NE. 0 ) THEN
            CALL MPI_SEND( BUFR(1,ISLAVE), TAILLE_SENDR,
     &                     MPI_COMPLEX, ISLAVE,
     &                     ARROWHEAD, COMM, IERR )
          END IF
        ENDDO
      RETURN
      END SUBROUTINE CMUMPS_ARROW_FINISH_SEND_BUF
      RECURSIVE SUBROUTINE CMUMPS_QUICK_SORT_ARROWHEADS( N, PERM, 
     &            INTLIST, DBLLIST, TAILLE, LO, HI )
      IMPLICIT NONE
      INTEGER N, TAILLE
      INTEGER PERM( N )
      INTEGER INTLIST( TAILLE )
      COMPLEX DBLLIST( TAILLE )
      INTEGER LO, HI
      INTEGER I,J
      INTEGER ISWAP, PIVOT
      COMPLEX cswap
      I = LO
      J = HI
      PIVOT = PERM(INTLIST((I+J)/2))
 10   IF (PERM(INTLIST(I)) < PIVOT) THEN
        I=I+1
        GOTO 10
      ENDIF
 20   IF (PERM(INTLIST(J)) > PIVOT) THEN
        J=J-1
        GOTO 20
      ENDIF
      IF (I < J) THEN
        ISWAP = INTLIST(I)
        INTLIST(I) = INTLIST(J)
        INTLIST(J)=ISWAP
        cswap = DBLLIST(I)
        DBLLIST(I) = DBLLIST(J)
        DBLLIST(J) = cswap
      ENDIF
      IF ( I <= J) THEN
        I = I+1
        J = J-1
      ENDIF
      IF ( I <= J ) GOTO 10
      IF ( LO < J ) CALL CMUMPS_QUICK_SORT_ARROWHEADS(N, PERM,
     &              INTLIST, DBLLIST, TAILLE, LO, J)
      IF ( I < HI ) CALL CMUMPS_QUICK_SORT_ARROWHEADS(N, PERM,
     &              INTLIST, DBLLIST, TAILLE, I, HI)
      RETURN
      END SUBROUTINE CMUMPS_QUICK_SORT_ARROWHEADS
      SUBROUTINE CMUMPS_FACTO_RECV_ARROWHD2(  N,
     &    DBLARR, LDBLARR, INTARR, LINTARR,
     &    PTR8ARR, NINCOLARR, NINROWARR, PTRDEBARR,
     &    KEEP, KEEP8, FILS, MYID,  COMM, NBRECORDS,
     &    A, LA, root, roota,
     &    PROCNODE_STEPS,
     &    SLAVEF, PERM, FRERE_STEPS, STEP, ICNTL, INFO )
      USE MUMPS_INTR_TYPES, ONLY : MUMPS_ROOT_STRUC
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_ROOT_STRUC
      IMPLICIT NONE
      INTEGER N, MYID, COMM
      INTEGER   KEEP(500)
      INTEGER(8), INTENT(IN) :: LDBLARR, LINTARR
      INTEGER INTARR(LINTARR) 
      INTEGER, INTENT(IN) :: FILS( N )
      INTEGER(8), INTENT(IN) :: PTR8ARR(KEEP(193))
      INTEGER, INTENT(IN) :: NINCOLARR(KEEP(194))
      INTEGER, INTENT(IN) :: NINROWARR(KEEP(195))
      INTEGER, INTENT(IN) :: PTRDEBARR(KEEP(196))
      INTEGER(8) KEEP8(150)
      INTEGER(8), intent(IN) :: LA
      INTEGER PROCNODE_STEPS( KEEP(28) ), PERM( N )
      INTEGER SLAVEF, NBRECORDS
      COMPLEX A( LA )
      INTEGER INFO( 80 ), ICNTL(60)
      COMPLEX DBLARR(LDBLARR)
      INTEGER FRERE_STEPS( KEEP(28) ), STEP(N)
      TYPE (MUMPS_ROOT_STRUC) :: root
      TYPE (CMUMPS_ROOT_STRUC) :: roota
      INTEGER LP
      LOGICAL LPOK
      INTEGER, ALLOCATABLE, DIMENSION(:) :: BUFI
      COMPLEX, ALLOCATABLE, DIMENSION(:) :: BUFR
      INTEGER, ALLOCATABLE, DIMENSION(:,:) :: IW4
      INTEGER(8), ALLOCATABLE, DIMENSION(:) :: PTRAW
      INTEGER :: IARR1, IORG, J, ISTEP
      LOGICAL :: EARLYT3ROOTINS
      LOGICAL FINI 
      INTEGER IREC, NB_REC, IARR, JARR, I, allocok
      INTEGER(8) :: IS8
      INTEGER LOCAL_M, LOCAL_N, ILOCROOT, JLOCROOT, 
     &        IPOSROOT, JPOSROOT, TAILLE,
     &        IPROC
      INTEGER(8) :: PTR_ROOT
      INTEGER ARROW_ROOT, TYPE_PARALL
      INTEGER MUMPS_TYPENODE, MUMPS_PROCNODE
      EXTERNAL MUMPS_TYPENODE, MUMPS_PROCNODE
      COMPLEX VAL
      COMPLEX ZERO
      PARAMETER( ZERO = (0.0E0,0.0E0) )
      INCLUDE 'mpif.h'
      INCLUDE 'mumps_tags.h'
      INTEGER MASTER
      PARAMETER(MASTER=0)
      INTEGER :: IERR
      INTEGER :: STATUS(MPI_STATUS_SIZE)
      TYPE_PARALL = KEEP(46)
      LP = ICNTL(1)
      LPOK = ( LP .GT. 0 .AND. ICNTL(4) .GE. 1 )
      ARROW_ROOT=0
      EARLYT3ROOTINS = KEEP(200) .EQ. 0
     & .OR. (KEEP(200) .LT. 0 .AND. KEEP(400) .EQ. 0)
      ALLOCATE( BUFI( NBRECORDS * 2 + 1 ), stat = allocok )
      IF ( allocok .GT. 0 ) THEN
        INFO(1) = -13
        INFO(2) = NBRECORDS * 2 + 1
        IF (LPOK) WRITE(LP,*) MYID,
     &  ': Error allocaing BUFI in CMUMPS_FACTO_RECV_ARROWHD2'
        GOTO 100
      END IF
      ALLOCATE( BUFR( NBRECORDS )        , stat = allocok )
      IF ( allocok .GT. 0 ) THEN
        INFO(1) = -13
        INFO(2) = NBRECORDS
        IF (LPOK) WRITE(LP,*) MYID,
     &  ': Error allocaing BUFR in CMUMPS_FACTO_RECV_ARROWHD2'
        GOTO 100
      END IF
      ALLOCATE( IW4(N,2), stat = allocok )
      IF ( allocok .GT. 0 ) THEN
        INFO(1) = -13
        CALL MUMPS_SET_IERROR( 2_8 * int(N,8), INFO(2) )
        IF (LPOK) WRITE(LP,*) MYID,
     &  ': Error allocaing IW4 in CMUMPS_FACTO_RECV_ARROWHD2'
        GOTO 100
      END IF
      ALLOCATE( PTRAW( N ), stat = allocok )
      IF ( allocok .GT. 0 ) THEN
        INFO(1) = -13
        INFO(2) = N
        IF (LPOK) WRITE(LP,*) MYID,
     &  ': Error allocaing PTRAW in CMUMPS_FACTO_RECV_ARROWHD2'
        GOTO 100
      END IF
 100  CONTINUE
      CALL MUMPS_PROPINFO( ICNTL, INFO, COMM, MYID )
      IF ( INFO( 1 ) .LT. 0 ) GOTO 500
      IF ( KEEP(38).NE.0 .AND. EARLYT3ROOTINS ) THEN
        CALL CMUMPS_GET_ROOT_INFO(root, LOCAL_M, LOCAL_N, PTR_ROOT, LA)
        CALL CMUMPS_SET_ROOT_TO_ZERO(root, roota, KEEP, A, LA)
      ELSE
        LOCAL_M = -19999; LOCAL_N = -29999; PTR_ROOT = -99999_8
      END IF
      FINI = .FALSE.
#if defined(__ve__)
!NEC$ IVDEP
#endif
      DO J = 1, N
        ISTEP=STEP(J)
        IF (ISTEP .GT. 0) THEN
          IARR1 = PTRDEBARR( ISTEP )
          IF ( IARR1 .GT. 0 ) THEN
            I = J
            IORG = 0
            DO WHILE ( I .GT. 0 )
              IORG = IORG + 1
              IW4(I, 1)  = NINCOLARR( IARR1 + IORG - 1 )
              IW4(I, 2)  = NINROWARR( IARR1 + IORG - 1 ) +
     &                     NINCOLARR( IARR1 + IORG - 1 )
              IS8        = PTR8ARR( IARR1 + IORG - 1 )
              PTRAW( I ) = IS8
              INTARR( IS8 ) = I
              DBLARR( IS8 ) = ZERO
              I = FILS(I)
            ENDDO
          ENDIF
        ENDIF
      ENDDO
      DO WHILE (.NOT.FINI) 
        CALL MPI_RECV( BUFI(1), 2*NBRECORDS+1, 
     &                MPI_INTEGER, MASTER, 
     &                ARROWHEAD,
     &                COMM, STATUS, IERR )
        NB_REC = BUFI(1)
        IF (NB_REC.LE.0) THEN
          FINI = .TRUE.
          NB_REC = -NB_REC 
        ENDIF
        IF (NB_REC.EQ.0) EXIT
        CALL MPI_RECV( BUFR(1), NBRECORDS, MPI_COMPLEX,
     &                  MASTER, ARROWHEAD,
     &                COMM, STATUS, IERR )
        DO IREC=1, NB_REC
          IARR = BUFI( IREC * 2 )
          JARR = BUFI( IREC * 2 + 1 )
          VAL  = BUFR( IREC )
          IF ( MUMPS_TYPENODE( PROCNODE_STEPS(abs(STEP(abs(IARR)))),
     &                         KEEP(199) ) .eq. 3
     &         .AND.  EARLYT3ROOTINS ) THEN
            IF ( IARR .GT. 0 ) THEN
              IPOSROOT = root%RG2L( IARR )
              JPOSROOT = root%RG2L( JARR )
            ELSE
              IPOSROOT = root%RG2L( JARR )
              JPOSROOT = root%RG2L( -IARR )
            END IF
            ILOCROOT = root%MBLOCK * ( ( IPOSROOT - 1 ) /
     &                ( root%MBLOCK * root%NPROW ) )
     &              + mod( IPOSROOT - 1, root%MBLOCK ) + 1
            JLOCROOT = root%NBLOCK * ( ( JPOSROOT - 1 ) /
     &                ( root%NBLOCK * root%NPCOL ) )
     &              + mod( JPOSROOT - 1, root%NBLOCK ) + 1
            IF (KEEP(60)==0) THEN
              A( PTR_ROOT + int(JLOCROOT - 1,8) * int(LOCAL_M,8)
     &                   + int(ILOCROOT - 1,8) )
     &        =  A( PTR_ROOT + int(JLOCROOT - 1,8)
     &                      * int(LOCAL_M,8)
     &                      + int(ILOCROOT - 1,8))
     &           + VAL
            ELSE
              roota%SCHUR_POINTER( int(JLOCROOT-1,8)
     &                          * int(root%SCHUR_LLD,8)
     &                          + int(ILOCROOT,8) )
     &        = roota%SCHUR_POINTER( int(JLOCROOT - 1,8)
     &                          * int(root%SCHUR_LLD,8)
     &                          + int(ILOCROOT,8))
     &          + VAL
            ENDIF
          ELSE IF (IARR.GE.0) THEN
            IF (IARR.EQ.JARR) THEN
              IS8         = PTRAW(IARR)
              DBLARR(IS8) = DBLARR( IS8 ) + VAL
            ELSE
              IS8         = PTRAW(IARR) + IW4(IARR,2)
              IW4(IARR,2) = IW4(IARR,2) - 1
              INTARR(IS8) = JARR
              DBLARR(IS8) = VAL
            ENDIF
          ELSE
            IARR        = -IARR
            IS8         = PTRAW(IARR)+IW4(IARR,1)
            IW4(IARR,1) = IW4(IARR,1) - 1
            INTARR(IS8) = JARR
            DBLARR(IS8) = VAL
            IF ( IW4(IARR,1) .EQ. 0 
     &         .AND. STEP(IARR) > 0 ) THEN
              IPROC = MUMPS_PROCNODE( PROCNODE_STEPS(STEP(IARR)),
     &                              KEEP(199) )
              IF ( TYPE_PARALL .eq. 0 ) THEN
                IPROC = IPROC + 1
              END IF 
              IF (IPROC .EQ. MYID) THEN
                TAILLE = NINCOLARR(PTRDEBARR(STEP( IARR ) ))
                CALL CMUMPS_QUICK_SORT_ARROWHEADS( N, PERM,
     &            INTARR( PTRAW(IARR) + 1 ),
     &            DBLARR( PTRAW(IARR) + 1 ),
     &            TAILLE, 1, TAILLE )
              END IF
            END IF
          ENDIF
        ENDDO
      END DO
 500  CONTINUE
      IF (allocated(BUFI  ) ) DEALLOCATE( BUFI  )
      IF (allocated(BUFR  ) ) DEALLOCATE( BUFR  )
      IF (allocated(IW4   ) ) DEALLOCATE( IW4   )
      IF (allocated(PTRAW ) ) DEALLOCATE( PTRAW )
      KEEP(49) = ARROW_ROOT
      RETURN 
      END SUBROUTINE CMUMPS_FACTO_RECV_ARROWHD2
      SUBROUTINE CMUMPS_SET_TO_ZERO(A, LLD, M, N, KEEP)
!$    USE OMP_LIB, ONLY : OMP_GET_MAX_THREADS
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: LLD, M, N
      COMPLEX             :: A(int(LLD,8)*int(N-1,8)+int(M,8))
      INTEGER             :: KEEP(500)
      COMPLEX, PARAMETER :: ZERO = (0.0E0,0.0E0)
      INTEGER I, J
!$    INTEGER :: NOMP
      INTEGER(8) :: I8, LA
!$    NOMP = OMP_GET_MAX_THREADS()
      IF (LLD .EQ. M) THEN
        LA=int(LLD,8)*int(N-1,8)+int(M,8)
!$OMP   PARALLEL DO PRIVATE(I8) SCHEDULE(STATIC,KEEP(361))
!$OMP&  IF ( LA > int(KEEP(361),8) .AND. NOMP .GT. 1)
        DO I8=1, LA
          A(I8) = ZERO
        ENDDO
!$OMP   END PARALLEL DO
      ELSE
!$OMP   PARALLEL DO PRIVATE(I,J) COLLAPSE(2)
!$OMP&  SCHEDULE(STATIC,KEEP(361)) IF (int(M,8)*int(N,8)
!$OMP&  .GT. KEEP(361).AND. NOMP .GT.1)
        DO I = 1, N
          DO J = 1, M
            A( int(I-1,8)*int(LLD,8)+ int(J,8) ) = ZERO
          ENDDO
        ENDDO
!$OMP   END PARALLEL DO
      ENDIF
      RETURN
      END SUBROUTINE CMUMPS_SET_TO_ZERO
      SUBROUTINE CMUMPS_SET_ROOT_TO_ZERO(root, roota, KEEP, A, LA)
      USE MUMPS_INTR_TYPES, ONLY : MUMPS_ROOT_STRUC
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_ROOT_STRUC
      IMPLICIT NONE
      INTEGER(8), INTENT(IN)   :: LA
      COMPLEX, INTENT(INOUT)   :: A(LA)
      INTEGER                  :: KEEP(500)
      TYPE (MUMPS_ROOT_STRUC)  :: root
      TYPE (CMUMPS_ROOT_STRUC) :: roota
      INTEGER :: LOCAL_M, LOCAL_N
      INTEGER(8) :: PTR_ROOT
      IF (KEEP(60)==0) THEN
        CALL CMUMPS_GET_ROOT_INFO(root, LOCAL_M, LOCAL_N, PTR_ROOT, LA)
        IF (LOCAL_N .GT. 0) THEN 
          CALL CMUMPS_SET_TO_ZERO(A(PTR_ROOT),
     &                            LOCAL_M, LOCAL_M, LOCAL_N, KEEP)
        ENDIF
      ELSE IF (root%yes) THEN
        CALL CMUMPS_SET_TO_ZERO(roota%SCHUR_POINTER(1),
     &       root%SCHUR_LLD, root%SCHUR_MLOC, root%SCHUR_NLOC,
     &       KEEP)
      ENDIF
      RETURN
      END SUBROUTINE CMUMPS_SET_ROOT_TO_ZERO
      SUBROUTINE CMUMPS_GET_ROOT_INFO(root,
     &                                LOCAL_M, LOCAL_N, PTR_ROOT, LA)
      USE MUMPS_INTR_TYPES, ONLY : MUMPS_ROOT_STRUC
      IMPLICIT NONE
      TYPE (MUMPS_ROOT_STRUC), INTENT(IN) :: root
      INTEGER,    INTENT(OUT) :: LOCAL_M, LOCAL_N
      INTEGER(8), INTENT(OUT) :: PTR_ROOT
      INTEGER(8), INTENT(IN)  :: LA
      INTEGER, EXTERNAL :: MUMPS_NUMROC
      LOCAL_M = MUMPS_NUMROC( root%ROOT_SIZE, root%MBLOCK,
     &         root%MYROW, 0, root%NPROW )
      LOCAL_M = max( 1, LOCAL_M )
      LOCAL_N = MUMPS_NUMROC( root%ROOT_SIZE, root%NBLOCK,
     &         root%MYCOL, 0, root%NPCOL )
      PTR_ROOT = LA - int(LOCAL_M,8) * int(LOCAL_N,8) + 1_8
      RETURN
      END SUBROUTINE CMUMPS_GET_ROOT_INFO
