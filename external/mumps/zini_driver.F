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
#if defined(__ve__)
#if defined(VHOFFLOAD)
#include 've.h'
#endif
#endif
      SUBROUTINE ZMUMPS_INI_DRIVER( id, idintr )
      USE ZMUMPS_STRUC_DEF
      USE ZMUMPS_INTR_TYPES, ONLY: ZMUMPS_INTR_STRUC
C
C  Purpose:
C  =======
C
C  Initialize an instance of the ZMUMPS package.
C
      IMPLICIT NONE
      INCLUDE 'mpif.h'
      TYPE (ZMUMPS_STRUC)      :: id
      TYPE (ZMUMPS_INTR_STRUC) :: idintr
      INTEGER MASTER, IERR,PAR_loc,SYM_loc
      PARAMETER( MASTER = 0 )
      INTEGER color
#if defined(metis) || defined(parmetis)
      INTEGER I
#endif
      INTEGER(8) :: I8
C     -----------------------------
C     Initialize MPI related data
C     -----------------------------
      CALL MPI_COMM_SIZE(id%COMM, id%NPROCS, IERR )
C     Now done in the main MUMPS driver:
C     CALL MPI_COMM_RANK(id%COMM, id%MYID, IERR )
C
      PAR_loc=id%PAR
      SYM_loc=id%SYM
C     Broadcasting PAR/SYM (KEEP(46)/KEEP(50)) in order to
C     have only one value available: the one from the master
      CALL MPI_BCAST(PAR_loc,1,MPI_INTEGER,MASTER,id%COMM,IERR)
      CALL MPI_BCAST(SYM_loc,1,MPI_INTEGER,MASTER,id%COMM,IERR)
C     Initialize a subcommunicator
C     for slave nodes
C
      IF ( PAR_loc .eq. 0 ) THEN
C       -------------------
C       Host is not working
C       -------------------
        IF ( id%MYID .eq. MASTER ) THEN
          color = MPI_UNDEFINED
        ELSE
          color = 0
        END IF
        CALL MPI_COMM_SPLIT( id%COMM, color, 0,
     &                       id%COMM_NODES, IERR )
        id%NSLAVES = id%NPROCS - 1
      ELSE
C       ----------------
C       Host is working
C       ----------------
        CALL MPI_COMM_DUP( id%COMM, id%COMM_NODES, IERR )
        id%NSLAVES = id%NPROCS
      END IF
C     ---------------------------
C     Use same slave communicator
C     for load information
C     ---------------------------
      IF (PAR_loc .ne. 0 .or. id%MYID .NE. MASTER) THEN
        CALL MPI_COMM_DUP( id%COMM_NODES, id%COMM_LOAD, IERR )
      ENDIF
C     ----------------------------------------------
C     Initialize default values for CNTL,ICNTL,KEEP,KEEP8
C     potentially depending on id%SYM and id%NSLAVES
C     ----------------------------------------------
      CALL ZMUMPSID( id%NSLAVES, id%LWK_USER,
     &    id%CNTL(1), id%ICNTL(1),
     &    id%KEEP(1), id%KEEP8(1), id%INFO(1), id%INFOG(1),
     &    id%RINFO(1), id%RINFOG(1),
     &    SYM_loc, PAR_loc, id%DKEEP(1), id%MYID )
      CALL MUMPS_BUILD_ARCH_NODE_COMM( id%COMM, id%KEEP(411),
     &     id%KEEP(412), id%KEEP(413), id%KEEP(410) )
      id%WRITE_PROBLEM="NAME_NOT_INITIALIZED"
      CALL MUMPS_SET_VERSION( id%VERSION_NUMBER )
      id%OOC_TMPDIR="NAME_NOT_INITIALIZED"
      id%OOC_PREFIX="NAME_NOT_INITIALIZED"
#if ! defined(NO_SAVE_RESTORE)
      id%SAVE_DIR="NAME_NOT_INITIALIZED"
      id%SAVE_PREFIX="NAME_NOT_INITIALIZED"
#endif
C     Default value for NRHS is 1
      id%NRHS = 1
C     Leading dimension will be reset to id%N is ZMUMPS_SOL_DRIVER
C     if id%NRHS remains equal to 1. Otherwise id%LRHS must be
C     set by user.
      id%LRHS = 0 ! Value will be checked in ZMUMPS_CHECK_DENSE_RHS
                  ! Not accessed if id%NRHS=1
C     Similar behaviour for LREDRHS (value will
C     be checked in ZMUMPS_CHECK_REDRHS)
      id%LREDRHS = 0
C
      id%INST_Number = -1
C
C     Define the options for Metis
C
      id%METIS_OPTIONS(:) = 0
#if defined(metis) || defined(parmetis) || defined(metis4) || defined(parmetis3)      
#if defined(metis4) || defined(parmetis3)
C     Useful size is 8
C     set to default options
      id%METIS_OPTIONS(1) = 0
#else
C     Useful size is 40  
C     This sets the default values
      CALL METIS_SETDEFAULTOPTIONS(id%METIS_OPTIONS)
      CALL MUMPS_METIS_OPTION_NUMBERING(I)
C     The value of I corresponds to "METIS_OPTION_NUMBERING", which tells
C     METIS to use Fortran numbering. METIS_OPTION_NUMBERING is defined
C     in metis.h and accessed through a C wrapper.
      id%METIS_OPTIONS(I+1) = 1 ! +1 for Fortran indexing
#endif
#endif      
C
C  Nullify a few pointers and integers
C
      id%N = 0; id%NZ = 0; id%NNZ = 0_8
      NULLIFY(id%IRN)
      NULLIFY(id%JCN)
      NULLIFY(id%A)
      id%NZ_loc = 0; id%NNZ_loc = 0_8
      NULLIFY(id%IRN_loc)
      NULLIFY(id%JCN_loc)
      NULLIFY(id%A_loc)
      NULLIFY(id%MAPPING)
      NULLIFY(id%RHS)
      NULLIFY(id%REDRHS)
      id%NZ_RHS=0
      NULLIFY(id%RHS_SPARSE)
      NULLIFY(id%IRHS_SPARSE)
      NULLIFY(id%IRHS_PTR)
      NULLIFY(id%ISOL_loc)
      NULLIFY(id%IRHS_loc)
      id%LSOL_loc=0
      id%LRHS_loc=0
      id%Nloc_RHS=0
      NULLIFY(id%SOL_loc)
      NULLIFY(id%RHS_loc)
      NULLIFY(id%COLSCA)
      NULLIFY(id%ROWSCA)
      NULLIFY(id%ROWSCA_loc)
      NULLIFY(id%COLSCA_loc)
      NULLIFY(id%PERM_IN)
      NULLIFY(id%IS)
      NULLIFY(id%STEP)
C     Info for analysis by block
      id%NBLK = 0
      NULLIFY(id%BLKPTR)
      NULLIFY(id%BLKVAR)
C     Info for pruning tree
      NULLIFY(id%Step2node)
      NULLIFY(id%DAD_STEPS)
      NULLIFY(id%NE_STEPS)
      NULLIFY(id%ND_STEPS)
      NULLIFY(id%FRERE_STEPS)
      NULLIFY(id%SYM_PERM)
      NULLIFY(id%UNS_PERM)
      NULLIFY(id%PIVNUL_LIST)
      NULLIFY(id%FILS)
      NULLIFY(id%PTRAR)
      NULLIFY(id%PTR8ARR)
      NULLIFY(id%NINCOLARR)
      NULLIFY(id%NINROWARR)
      NULLIFY(id%PTRDEBARR)
      NULLIFY(id%FRTPTR)
      NULLIFY(id%FRTELT)
      NULLIFY(id%NA)
      id%LNA=0
      NULLIFY(id%PROCNODE_STEPS)
      NULLIFY(id%S)
      NULLIFY(id%LPS)
      NULLIFY(id%PTLUST_S)
      NULLIFY(id%PTRFAC)
      NULLIFY(id%DEPTH_FIRST)
      NULLIFY(id%DEPTH_FIRST_SEQ)
      NULLIFY(id%SBTR_ID)
      NULLIFY(id%SCHED_DEP)
      NULLIFY(id%SCHED_SBTR)
      NULLIFY(id%SCHED_GRP)
      NULLIFY(id%CROIX_MANU)
      NULLIFY(id%WK_USER)
      NULLIFY(id%MEM_SUBTREE)
      NULLIFY(id%MEM_SUBTREE)
      NULLIFY(id%MY_ROOT_SBTR)
      NULLIFY(id%MY_FIRST_LEAF)
      NULLIFY(id%MY_NB_LEAF)
      NULLIFY(id%COST_TRAV)
      NULLIFY(id%RHSINTR)
      id%LD_RHSINTR = 0
      NULLIFY(id%GLOB2LOC_RHS)
      NULLIFY(id%GLOB2LOC_SOL)
      id%GLOB2LOC_SOL_ALLOC = .FALSE.
C
C     Out of Core management related data
C
      NULLIFY(id%OOC_INODE_SEQUENCE)
      NULLIFY(id%OOC_TOTAL_NB_NODES)
      NULLIFY(id%OOC_SIZE_OF_BLOCK)
      NULLIFY(id%OOC_FILE_NAME_LENGTH)
      NULLIFY(id%OOC_FILE_NAMES)
      NULLIFY(id%OOC_VADDR)
      NULLIFY(id%OOC_NB_FILES)
      NULLIFY(id%LRGROUPS)
      NULLIFY(id%FDM_F_ENCODING)
      NULLIFY(id%BLRARRAY_ENCODING)
      NULLIFY(id%MTKO_PROCS_MAP)
C     Must be nullified because of routine
C     ZMUMPS_SIZE_IN_STRUCT
      NULLIFY(id%CB_SON_SIZE)
C
C     Components of the arithmetic-dependent root
C
      CALL ZMUMPS_INI_ROOT(idintr%roota)
      NULLIFY(idintr%root%RG2L)
      NULLIFY(idintr%root%IPIV)
      NULLIFY(id%SCHUR_CINTERFACE)
C
C     Element-entry
C
      id%NELT=0
      NULLIFY(id%ELTPTR)
      NULLIFY(id%ELTVAR)
      NULLIFY(id%A_ELT)
      NULLIFY(id%ELTPROC)
C
C     Schur
C
      id%SIZE_SCHUR = 0
      NULLIFY( id%LISTVAR_SCHUR )
      NULLIFY( id%SCHUR )
C     -- Distributed Schur
      id%NPROW      = 0
      id%NPCOL      = 0
      id%MBLOCK     = 0
      id%NBLOCK     = 0
      id%SCHUR_MLOC = 0 ! Exit from analysis
      id%SCHUR_NLOC = 0 ! Exit from analysis
      id%SCHUR_LLD  = 0
C
C     Candidates and node partitionning
C
      NULLIFY(id%ISTEP_TO_INIV2)
      NULLIFY(id%I_AM_CAND)
      NULLIFY(id%FUTURE_NIV2)
      NULLIFY(id%TAB_POS_IN_PERE)
      NULLIFY(id%CANDIDATES)
      id%OOC_NB_FILE_TYPE=-123456
C
C     Initializations for L0_OMP mechanisms
C
      NULLIFY(id%IPOOL_B_L0_OMP)
      NULLIFY(id%IPOOL_A_L0_OMP)
      NULLIFY(id%PHYS_L0_OMP)
      NULLIFY(id%VIRT_L0_OMP)
      NULLIFY(id%VIRT_L0_OMP_MAPPING)
      NULLIFY(id%PERM_L0_OMP)
      NULLIFY(id%PTR_LEAFS_L0_OMP)
      NULLIFY(id%L0_OMP_MAPPING)
      NULLIFY(idintr%L0_OMP_FACTORS)
      NULLIFY(id%I4_L0_OMP)
      NULLIFY(id%I8_L0_OMP)
      id%LPOOL_B_L0_OMP = 0
      id%LPOOL_A_L0_OMP  = 0
      id%L_VIRT_L0_OMP       = 0
      id%L_PHYS_L0_OMP       = 0
      id%THREAD_LA           = 0
C
C     Mapping information used during solve.
C
      NULLIFY(id%IPTR_WORKING)
      NULLIFY(id%WORKING)
C
C     Initializations for Rank detection/null space
C
      NULLIFY(id%SINGULAR_VALUES)
C     Architecture data
      NULLIFY(id%MEM_DIST)
C     Must be nullified because of routine
C     ZMUMPS_SIZE_IN_STRUCT
      NULLIFY(id%SUP_PROC)
      id%Deficiency = 0
      idintr%root%LPIV = -1
      idintr%root%yes  = .FALSE.
      idintr%root%gridinit_done  = .FALSE.
C     NOT IN SAVE/RESTORE
      id%ASSOCIATED_OOC_FILES=.FALSE.
C
C     ----------------------------------------
C     Find MYID_NODES relatively to COMM_NODES
C     If  the calling processor is not inside
C     COMM_NODES, MYID_NODES will not be
C     significant / used anyway
C     ----------------------------------------
      IF ( id%KEEP( 46 ) .ne. 0  .OR.
     &     id%MYID .ne. MASTER ) THEN
          CALL MPI_COMM_RANK
     &         (id%COMM_NODES, id%MYID_NODES, IERR )
      ELSE
          id%MYID_NODES = -464646
      ENDIF
C
C     Check that KEEP(34), the size of a Fortran INTEGER,
C     as initialized above during ZMUMPSID
C     matches the size of an integer in C. If not,
C     raise an error immediately.
C
      CALL MUMPS_INT_SIZE_C(I8)
      IF (int(I8) .NE. id%KEEP(34)) THEN
        id%INFO(1)=-69
        id%INFO(2)=int(I8) ! size of MUMPS_INT
C       Installation problem!
C       WRITE on unit 6 since ICNTL(1:4) are not set by the user yet
        IF (id%MYID .EQ. 0) WRITE(6,995) int(I8)
 995    FORMAT(' Installation error -69: ',
     &' MUMPS_INT size (',I4,') incompatible with INTEGER size')
      ENDIF
      RETURN
      END SUBROUTINE ZMUMPS_INI_DRIVER
      SUBROUTINE ZMUMPS_INI_ROOT(roota)
      USE ZMUMPS_INTR_TYPES, ONLY: ZMUMPS_ROOT_STRUC
      IMPLICIT NONE
      TYPE(ZMUMPS_ROOT_STRUC) :: roota
      NULLIFY(roota%RHS_CNTR_MASTER_ROOT)
      NULLIFY(roota%RHS_ROOT)
      NULLIFY(roota%SCHUR_POINTER)
      CALL ZMUMPS_RR_INIT_POINTERS(roota)
      RETURN
      END SUBROUTINE ZMUMPS_INI_ROOT
