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
      MODULE CMUMPS_FACSOL_L0OMP_M
      PRIVATE
      PUBLIC :: CMUMPS_INIT_L0_OMP_FACTORS
     &        , CMUMPS_FREE_L0_OMP_FACTORS
#if ! defined(NO_SAVE_RESTORE)
     &        , CMUMPS_SAVE_RESTORE_L0FACARRAY
#endif
#if ! defined(NO_SAVE_RESTORE)
#endif
#if ! defined(NO_SAVE_RESTORE)
      INCLUDE 'mumps_save_restore_modes.h'
#endif
      CONTAINS
      SUBROUTINE CMUMPS_INIT_L0_OMP_FACTORS(id_L0_OMP_FACTORS)
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_L0OMPFAC_T
      IMPLICIT NONE
      TYPE (CMUMPS_L0OMPFAC_T), DIMENSION(:), POINTER ::
     &                                             id_L0_OMP_FACTORS
      INTEGER I
      IF (associated(id_L0_OMP_FACTORS)) THEN
        DO I=1, size(id_L0_OMP_FACTORS)
          NULLIFY(id_L0_OMP_FACTORS(I)%A)
        ENDDO
      ENDIF
      RETURN
      END SUBROUTINE CMUMPS_INIT_L0_OMP_FACTORS
      SUBROUTINE CMUMPS_FREE_L0_OMP_FACTORS(id_L0_OMP_FACTORS)
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_L0OMPFAC_T
      IMPLICIT NONE
      TYPE (CMUMPS_L0OMPFAC_T), DIMENSION(:), POINTER ::
     &                                              id_L0_OMP_FACTORS
      INTEGER I
      IF (associated(id_L0_OMP_FACTORS)) THEN
        DO I=1, size(id_L0_OMP_FACTORS)
          IF (associated(id_L0_OMP_FACTORS(I)%A)) THEN
            DEALLOCATE(id_L0_OMP_FACTORS(I)%A)
            NULLIFY(id_L0_OMP_FACTORS(I)%A)
          ENDIF
        ENDDO
        DEALLOCATE(id_L0_OMP_FACTORS)
        NULLIFY(id_L0_OMP_FACTORS)
      ENDIF
      RETURN
      END SUBROUTINE CMUMPS_FREE_L0_OMP_FACTORS
#if ! defined(NO_SAVE_RESTORE)
      SUBROUTINE CMUMPS_SAVE_RESTORE_L0FACARRAY(L0_OMP_FACTORS
     &                 ,unit,MYID,mode
     &                 ,SIZE_GEST,SIZE_VARIABLES
     &                 ,SIZE_INT,SIZE_INT8,SIZE_ARITH_DEP
     &                 ,TOTAL_FILE_SIZE,TOTAL_STRUC_SIZE
     &                 ,size_read,size_allocated,size_written
     &                 ,INFO)
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_L0OMPFAC_T
      IMPLICIT NONE
      TYPE (CMUMPS_L0OMPFAC_T), DIMENSION(:), POINTER :: L0_OMP_FACTORS
      INTEGER,intent(IN)::unit,MYID
      INTEGER,intent(IN) :: mode
      INTEGER,INTENT(OUT) :: SIZE_GEST
      INTEGER(8),intent(OUT) :: SIZE_VARIABLES
      INTEGER,intent(INOUT):: INFO(2)
      INTEGER,intent(IN):: SIZE_INT, SIZE_INT8, SIZE_ARITH_DEP
      INTEGER(8),intent(IN) :: TOTAL_FILE_SIZE,TOTAL_STRUC_SIZE
      INTEGER(8),intent(INOUT):: size_read,size_allocated,size_written
      INTEGER:: j1,NbRecords,NbSubRecords,size_array1,dummy,allocok,err
      INTEGER:: SIZE_GEST_L0FAC_ARRAY,
     &          SIZE_GEST_L0FAC_ARRAY_j1
      INTEGER(4) :: I4
      INTEGER(8):: SIZE_VARIABLES_L0FAC_ARRAY,
     &             SIZE_VARIABLES_L0FAC_ARRAY_j1
      SIZE_GEST = 0
      SIZE_VARIABLES = 0_8
      SIZE_GEST_L0FAC_ARRAY=0
      SIZE_VARIABLES_L0FAC_ARRAY=0
      SIZE_GEST_L0FAC_ARRAY_j1=0
      SIZE_VARIABLES_L0FAC_ARRAY_j1=0
      NbRecords = 0
      IF (mode.EQ.memory_save_mode) THEN
        IF (associated(L0_OMP_FACTORS)) THEN
          NbRecords      = 1 
          SIZE_GEST      = SIZE_INT 
          SIZE_VARIABLES = 0
          DO j1=1,size(L0_OMP_FACTORS)
            CALL CMUMPS_SAVE_RESTORE_L0FAC(
     &           L0_OMP_FACTORS(j1)
     &           ,unit,MYID,memory_save_mode
     &           ,SIZE_GEST_L0FAC_ARRAY_j1
     &           ,SIZE_VARIABLES_L0FAC_ARRAY_J1
     &           ,SIZE_INT,SIZE_INT8,SIZE_ARITH_DEP
     &           ,TOTAL_FILE_SIZE,TOTAL_STRUC_SIZE
     &           ,size_read,size_allocated,size_written
     &           ,INFO)
            SIZE_GEST_L0FAC_ARRAY=SIZE_GEST_L0FAC_ARRAY+
     &           SIZE_GEST_L0FAC_ARRAY_j1
            SIZE_VARIABLES_L0FAC_ARRAY=SIZE_VARIABLES_L0FAC_ARRAY+
     &           SIZE_VARIABLES_L0FAC_ARRAY_j1
            IF ( INFO(1) .LT. 0 ) GOTO 100
          ENDDO
        ELSE
          NbRecords      = 2
          SIZE_GEST      = 2*SIZE_INT
          SIZE_VARIABLES = 0
        ENDIF
      ELSEIF (mode.EQ.save_mode) THEN
        IF (associated(L0_OMP_FACTORS)) THEN
          NbRecords      = 1 
          SIZE_GEST      = SIZE_INT
          SIZE_VARIABLES = 0
          write(unit,iostat=err) size(L0_OMP_FACTORS)
          if(err.ne.0) then
             INFO(1) = -72
             CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_written,
     &            INFO(2))
          endif
          IF ( INFO(1) .LT. 0 ) GOTO 100
          DO j1=1,size(L0_OMP_FACTORS)
            CALL CMUMPS_SAVE_RESTORE_L0FAC(
     &           L0_OMP_FACTORS(j1)
     &           ,unit,MYID,save_mode
     &           ,SIZE_GEST_L0FAC_ARRAY_J1
     &           ,SIZE_VARIABLES_L0FAC_ARRAY_J1
     &           ,SIZE_INT,SIZE_INT8,SIZE_ARITH_DEP
     &           ,TOTAL_FILE_SIZE,TOTAL_STRUC_SIZE
     &           ,size_read,size_allocated,size_written
     &           ,INFO)
          ENDDO
        ELSE
            NbRecords=2
            SIZE_GEST=SIZE_INT*2
            SIZE_VARIABLES=0
            write(unit,iostat=err) -999
            if(err.ne.0) then
               INFO(1) = -72
               CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_written,
     &              INFO(2))
            endif
            IF ( INFO(1) .LT. 0 ) GOTO 100
            write(unit,iostat=err) -999
            if(err.ne.0) then
               INFO(1) = -72
               CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_written,
     &              INFO(2))
            endif
            IF ( INFO(1) .LT. 0 ) GOTO 100
        ENDIF
      ELSE IF (mode.EQ.restore_mode) THEN
        NULLIFY(L0_OMP_FACTORS)
         read(unit,iostat=err) size_array1
         if(err.ne.0) THEN
            INFO(1) = -75
            CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_read
     &           ,INFO(2))
         endif
         IF ( INFO(1) .LT. 0 ) GOTO 100
         if(size_array1.EQ.-999) then
            NbRecords=2
            SIZE_GEST=SIZE_INT*2
            SIZE_VARIABLES=0
            read(unit,iostat=err) dummy
            if(err.ne.0) THEN
               INFO(1) = -75
               CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_read
     &              ,INFO(2))
            endif
            IF ( INFO(1) .LT. 0 ) GOTO 100
         else
           NbRecords=1
           SIZE_GEST=SIZE_INT
           SIZE_VARIABLES=0
           allocate(L0_OMP_FACTORS(size_array1), stat=allocok)
           if (allocok .GT. 0) THEN
              INFO(1) = -78
              CALL MUMPS_SETI8TOI4(
     &             TOTAL_STRUC_SIZE-size_allocated
     &             ,INFO(2))
           endif
           DO j1=1,size(L0_OMP_FACTORS)
             CALL CMUMPS_SAVE_RESTORE_L0FAC(
     &            L0_OMP_FACTORS(j1)
     &            ,unit,MYID,restore_mode
     &            ,SIZE_GEST_L0FAC_ARRAY_J1
     &            ,SIZE_VARIABLES_L0FAC_ARRAY_J1
     &            ,SIZE_INT, SIZE_INT8, SIZE_ARITH_DEP
     &            ,TOTAL_FILE_SIZE,TOTAL_STRUC_SIZE
     &            ,size_read,size_allocated,size_written
     &            ,INFO)
             SIZE_GEST_L0FAC_ARRAY=SIZE_GEST_L0FAC_ARRAY+
     &            SIZE_GEST_L0FAC_ARRAY_j1
             SIZE_VARIABLES_L0FAC_ARRAY=SIZE_VARIABLES_L0FAC_ARRAY+
     &            SIZE_VARIABLES_L0FAC_ARRAY_j1
             IF ( INFO(1) .LT. 0 ) GOTO 100
           ENDDO
         endif
      ENDIF
      if(mode.EQ.memory_save_mode) then
         NbSubRecords=int(SIZE_VARIABLES/huge(I4))
         IF(NbSubRecords.GT.0) then
            NbRecords=NbRecords+NbSubRecords
         ENDIF
      elseif(mode.EQ.save_mode) then
         size_written=size_written+SIZE_VARIABLES
     &        +int(SIZE_GEST,kind=8)
#if defined(MUMPS_NOF2003)
         size_written=size_written
     &        +int(2*SIZE_INT*NbRecords,kind=8)
#endif
      elseif(mode.EQ.restore_mode) then
         size_allocated=size_allocated+SIZE_VARIABLES
         size_read=size_read+SIZE_VARIABLES
     &        +int(SIZE_GEST,kind=8)
#if defined(MUMPS_NOF2003)
         size_read=size_read
     &        +int(2*SIZE_INT*NbRecords,kind=8)
#endif
      endif
      if(mode.EQ.memory_save_mode) then
         SIZE_VARIABLES=SIZE_VARIABLES+SIZE_VARIABLES_L0FAC_ARRAY
         SIZE_GEST=SIZE_GEST+SIZE_GEST_L0FAC_ARRAY
#if defined(MUMPS_NOF2003)
         SIZE_GEST=SIZE_GEST+2*SIZE_INT*NbRecords
#endif
      endif
 100  continue
      RETURN
      END SUBROUTINE CMUMPS_SAVE_RESTORE_L0FACARRAY
      SUBROUTINE CMUMPS_SAVE_RESTORE_L0FAC(
     &            L0_OMP_FACTORS_1THREAD
     &            ,unit,MYID,mode
     &            ,Local_SIZE_GEST, Local_SIZE_VARIABLES
     &            ,SIZE_INT, SIZE_INT8, SIZE_ARITH_DEP
     &            ,TOTAL_FILE_SIZE,TOTAL_STRUC_SIZE
     &            ,size_read,size_allocated,size_written
     &            ,INFO)
      USE CMUMPS_INTR_TYPES, ONLY : CMUMPS_L0OMPFAC_T
      IMPLICIT NONE
      TYPE (CMUMPS_L0OMPFAC_T) :: L0_OMP_FACTORS_1THREAD
      INTEGER,intent(IN)::unit,MYID
      INTEGER,intent(IN) :: mode
      INTEGER,INTENT(OUT) :: Local_SIZE_GEST
      INTEGER(8),intent(OUT) :: Local_SIZE_VARIABLES
      INTEGER,intent(INOUT):: INFO(2)
      INTEGER,intent(IN):: SIZE_INT, SIZE_INT8, SIZE_ARITH_DEP
      INTEGER(8),intent(IN) :: TOTAL_FILE_SIZE,TOTAL_STRUC_SIZE
      INTEGER(8),intent(INOUT):: size_read,size_allocated,size_written
      INTEGER:: Local_NbRecords, allocok, err
      INTEGER(8) :: itmp
      Local_NbRecords = 0
      Local_SIZE_GEST = 0
      Local_SIZE_VARIABLES = 0_8
      Local_NbRecords = Local_NbRecords+1
      IF (mode .EQ. memory_save_mode) THEN
        Local_SIZE_VARIABLES = Local_SIZE_VARIABLES + SIZE_INT8
      ELSE IF (mode .EQ. save_mode) THEN
        Local_SIZE_VARIABLES = Local_SIZE_VARIABLES + SIZE_INT8
        WRITE(unit,iostat=err) L0_OMP_FACTORS_1THREAD%LA
        IF (err .NE. 0) THEN
          INFO(1)=-72
          CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_written,
     &                         INFO(2))
          GOTO 100
        ENDIF
        size_written=size_written+SIZE_INT8
      ELSE IF (mode .EQ. restore_mode) THEN
        Local_SIZE_VARIABLES = Local_SIZE_VARIABLES + SIZE_INT8
        READ(unit,iostat=err) L0_OMP_FACTORS_1THREAD%LA
        IF (err .NE. 0) THEN
          INFO(1) = -75
          CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_read,
     &                         INFO(2))
          GOTO 100
        ENDIF
        size_read=size_read+SIZE_INT8
      ENDIF
      IF (mode.EQ.memory_save_mode) THEN
        IF (associated(L0_OMP_FACTORS_1THREAD%A)) THEN
          Local_NbRecords = Local_NbRecords + 2
          Local_SIZE_GEST = Local_SIZE_GEST + SIZE_INT8
          Local_SIZE_VARIABLES = Local_SIZE_VARIABLES + 
     &           max(1_8,L0_OMP_FACTORS_1THREAD%LA)*SIZE_ARITH_DEP
        ELSE
          Local_NbRecords = Local_NbRecords + 1
          Local_SIZE_GEST = Local_SIZE_GEST + SIZE_INT8
          Local_SIZE_VARIABLES = Local_SIZE_VARIABLES + 0
        ENDIF
      ELSEIF (mode.EQ.save_mode) THEN
        IF (associated(L0_OMP_FACTORS_1THREAD%A)) THEN
          Local_NbRecords = Local_NbRecords + 2
          write(unit,iostat=err) int(0,8) 
          if(err.ne.0) then
            INFO(1) = -72
            CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_written,
     &                    INFO(2))
            GOTO 100
          endif
          size_written=size_written+SIZE_INT8
          write(unit,iostat=err) L0_OMP_FACTORS_1THREAD%A
          if(err.ne.0) then
            INFO(1) = -72
            CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_written,
     &                    INFO(2))
            GOTO 100
          endif
          size_written = size_written +
     &              max(L0_OMP_FACTORS_1THREAD%LA,1_8)*SIZE_ARITH_DEP
        ELSE
          Local_NbRecords = Local_NbRecords + 1
          write(unit,iostat=err) int(-999,8)
          if(err.ne.0) then
            INFO(1) = -72
            CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_written,
     &                    INFO(2))
            GOTO 100
          endif
          size_written=size_written+SIZE_INT8
        ENDIF
      ELSEIF (mode.EQ.restore_mode) THEN
        NULLIFY(L0_OMP_FACTORS_1THREAD%A)
        READ(unit,iostat=err) itmp
        if(err.ne.0) THEN
          INFO(1) = -75
          CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_read
     &         ,INFO(2))
          GOTO 100
        endif
        size_read      = size_read      + SIZE_INT8
        size_allocated = size_allocated + SIZE_INT8
        IF (itmp .eq. -999) THEN
          Local_NbRecords = Local_NbRecords + 1
        ELSE
          Local_NbRecords = Local_NbRecords + 2
          ALLOCATE(L0_OMP_FACTORS_1THREAD%A(
     &             max(L0_OMP_FACTORS_1THREAD%LA,1_8)),
     &             stat=allocok)
          IF (allocok .GT. 0) THEN
            INFO(1) = -78
            CALL MUMPS_SETI8TOI4(
     &           TOTAL_STRUC_SIZE-size_allocated
     &           ,INFO(2))
            GOTO 100
          ENDIF
          READ(unit,iostat=err) L0_OMP_FACTORS_1THREAD%A
          if(err.ne.0) THEN
            INFO(1) = -75
            CALL MUMPS_SETI8TOI4(TOTAL_FILE_SIZE-size_read
     &           ,INFO(2))
            GOTO 100
          endif
          size_read      = size_read +
     &             max(1_8,L0_OMP_FACTORS_1THREAD%LA)*SIZE_ARITH_DEP
          size_allocated = size_allocated+
     &             max(1_8,L0_OMP_FACTORS_1THREAD%LA)*SIZE_ARITH_DEP
        ENDIF
      ENDIF
#if defined(MUMPS_NOF2003)
      IF (mode.EQ.memory_save_mode) THEN
        Local_SIZE_GEST = Local_SIZE_GEST+2*SIZE_INT*Local_NbRecords
      ELSE IF (mode.EQ.save_mode) THEN
        size_written = size_written+2*SIZE_INT*Local_NbRecords
      ELSE IF (mode.EQ.restore_mode) THEN
        size_read = size_read+2*SIZE_INT*Local_NbRecords
      ENDIF
#endif
  100 CONTINUE
      RETURN
      END SUBROUTINE CMUMPS_SAVE_RESTORE_L0FAC
#endif
      END MODULE CMUMPS_FACSOL_L0OMP_M
