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
      MODULE DMUMPS_TPS_M
      TYPE DMUMPS_TPS_T
          DOUBLE PRECISION, DIMENSION(:), POINTER :: A
      END TYPE DMUMPS_TPS_T
      END MODULE DMUMPS_TPS_M
      SUBROUTINE DMUMPS_TPS_M_RETURN()
      RETURN
      END SUBROUTINE DMUMPS_TPS_M_RETURN
