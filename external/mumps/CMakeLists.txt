cmake_minimum_required(VERSION 3.21)

project(mumps C Fortran)

add_compile_options(-w)

add_compile_definitions(Add_)
add_compile_definitions(pord)
add_compile_definitions(metis)
add_compile_definitions(parmetis)

include_directories(../gk/include)

if (MKL_INTERFACE STREQUAL ilp64)
    set(MKL_INTERFACE lp64)
endif ()
find_package(MKL QUIET PATHS "/opt/intel/oneapi/mkl/latest")
if (MKL_FOUND)
    add_compile_definitions(GEMMT_AVAILABLE)
endif ()

if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -w -fallow-argument-mismatch -fno-strict-aliasing --param=max-vartrack-size=500000000")
elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp -diag-disable=10440")
endif ()

file(GLOB mumps_f "*.F")
file(GLOB mumps_c "*.c")

set(mumps_mod
        ana_blk_m.F
        ana_omp_m.F
        ana_orderings_wrappers_m.F
        double_linked_list.F
        fac_asm_build_sort_index_ELT_m.F
        fac_asm_build_sort_index_m.F
        fac_descband_data_m.F
        fac_future_niv2_mod.F
        fac_maprow_data_m.F
        front_data_mgt_m.F
        lr_common.F
        lr_stats.F
        mumps_comm_buffer_common.F
        mumps_intr_types_common.F
        mumps_l0_omp_m.F
        mumps_load.F
        mumps_memory_mod.F
        mumps_mpitoomp_m.F
        mumps_ooc_common.F
        mumps_pivnul_mod.F
        mumps_static_mapping.F
        omp_tps_common_m.F
        sol_ds_common_m.F
        sol_omp_common_m.F
        tools_common_m.F

        dana_aux_par.F
        dana_aux.F
        dana_lr.F
        dfac_asm_master_ELT_m.F
        dfac_asm_master_m.F
        dfac_compact_factors_m.F
        dfac_front_aux.F
        dfac_front_LDLT_type1.F
        dfac_front_LDLT_type2.F
        dfac_front_LU_type1.F
        dfac_front_LU_type2.F
        dfac_front_type2_aux.F
        dfac_lr.F
        dfac_mem_dynamic.F
        dfac_omp_m.F
        dfac_par_m.F
        dfac_sispointers_m.F
        dfac_sol_l0omp_m.F
        dlr_core.F
        dlr_type.F
        dmumps_comm_buffer.F
        dmumps_config_file.F
        dmumps_intr_types.F
        dmumps_lr_data_m.F
        dmumps_mpi3_mod.F
        dmumps_ooc_buffer.F
        dmumps_ooc.F
        dmumps_save_restore_files.F
        dmumps_save_restore.F
        dmumps_sol_es.F
        dmumps_struc_def.F
        domp_tps_m.F
        dsol_lr.F
        dsol_omp_m.F
        dstatic_ptr_m.F

        sana_aux_par.F
        sana_aux.F
        sana_lr.F
        sfac_asm_master_ELT_m.F
        sfac_asm_master_m.F
        sfac_compact_factors_m.F
        sfac_front_aux.F
        sfac_front_LDLT_type1.F
        sfac_front_LDLT_type2.F
        sfac_front_LU_type1.F
        sfac_front_LU_type2.F
        sfac_front_type2_aux.F
        sfac_lr.F
        sfac_mem_dynamic.F
        sfac_omp_m.F
        sfac_par_m.F
        sfac_sispointers_m.F
        sfac_sol_l0omp_m.F
        slr_core.F
        slr_type.F
        smumps_comm_buffer.F
        smumps_config_file.F
        smumps_intr_types.F
        smumps_lr_data_m.F
        smumps_mpi3_mod.F
        smumps_ooc_buffer.F
        smumps_ooc.F
        smumps_save_restore_files.F
        smumps_save_restore.F
        smumps_sol_es.F
        smumps_struc_def.F
        somp_tps_m.F
        ssol_lr.F
        ssol_omp_m.F
        sstatic_ptr_m.F

        zana_aux_par.F
        zana_aux.F
        zana_lr.F
        zfac_asm_master_ELT_m.F
        zfac_asm_master_m.F
        zfac_compact_factors_m.F
        zfac_front_aux.F
        zfac_front_LDLT_type1.F
        zfac_front_LDLT_type2.F
        zfac_front_LU_type1.F
        zfac_front_LU_type2.F
        zfac_front_type2_aux.F
        zfac_lr.F
        zfac_mem_dynamic.F
        zfac_omp_m.F
        zfac_par_m.F
        zfac_sispointers_m.F
        zfac_sol_l0omp_m.F
        zlr_core.F
        zlr_type.F
        zmumps_comm_buffer.F
        zmumps_config_file.F
        zmumps_intr_types.F
        zmumps_lr_data_m.F
        zmumps_mpi3_mod.F
        zmumps_ooc_buffer.F
        zmumps_ooc.F
        zmumps_save_restore_files.F
        zmumps_save_restore.F
        zmumps_sol_es.F
        zmumps_struc_def.F
        zomp_tps_m.F
        zsol_lr.F
        zsol_omp_m.F
        zstatic_ptr_m.F

        cana_aux_par.F
        cana_aux.F
        cana_lr.F
        cfac_asm_master_ELT_m.F
        cfac_asm_master_m.F
        cfac_compact_factors_m.F
        cfac_front_aux.F
        cfac_front_LDLT_type1.F
        cfac_front_LDLT_type2.F
        cfac_front_LU_type1.F
        cfac_front_LU_type2.F
        cfac_front_type2_aux.F
        cfac_lr.F
        cfac_mem_dynamic.F
        cfac_omp_m.F
        cfac_par_m.F
        cfac_sispointers_m.F
        cfac_sol_l0omp_m.F
        clr_core.F
        clr_type.F
        cmumps_comm_buffer.F
        cmumps_config_file.F
        cmumps_intr_types.F
        cmumps_lr_data_m.F
        cmumps_mpi3_mod.F
        cmumps_ooc_buffer.F
        cmumps_ooc.F
        cmumps_save_restore_files.F
        cmumps_save_restore.F
        cmumps_sol_es.F
        cmumps_struc_def.F
        comp_tps_m.F
        csol_lr.F
        csol_omp_m.F
        cstatic_ptr_m.F
)

foreach (mod_src ${mumps_mod})
    foreach (f_src ${mumps_f})
        if (f_src MATCHES ${mod_src})
            list(REMOVE_ITEM mumps_f ${f_src})
            break()
        endif ()
    endforeach ()
endforeach ()

set_source_files_properties(dmumps_c.c PROPERTIES COMPILE_DEFINITIONS "MUMPS_ARITH=MUMPS_ARITH_d")
set_source_files_properties(smumps_c.c PROPERTIES COMPILE_DEFINITIONS "MUMPS_ARITH=MUMPS_ARITH_s")
set_source_files_properties(zmumps_c.c PROPERTIES COMPILE_DEFINITIONS "MUMPS_ARITH=MUMPS_ARITH_z")
set_source_files_properties(cmumps_c.c PROPERTIES COMPILE_DEFINITIONS "MUMPS_ARITH=MUMPS_ARITH_c")

add_library(${PROJECT_NAME} STATIC ${mumps_c} ${mumps_f} ${mumps_mod})
if (MKL_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE MKL::MKL_SCALAPACK)
elseif (EZP_USE_SYSTEM_LIBS)
    set(EXT_LIBS scalapack flexiblas openblas)
    foreach (EXT_LIB ${EXT_LIBS})
        find_library(${EXT_LIB}_LIB ${EXT_LIB})
        if (${EXT_LIB}_LIB)
            target_link_libraries(${PROJECT_NAME} PRIVATE ${${EXT_LIB}_LIB})
        endif ()
    endforeach ()
else ()
    link_directories(${PROJECT_SOURCE_DIR}/../libs)
    if (MPI_VERSION_FILE MATCHES "mpichversion")
        target_link_libraries(${PROJECT_NAME} PRIVATE scalapack-mpich)
    elseif (MPI_VERSION_FILE MATCHES "ompi_info")
        target_link_libraries(${PROJECT_NAME} PRIVATE scalapack-openmpi)
    endif ()
    target_link_libraries(${PROJECT_NAME} PRIVATE lapack blas)
endif ()
target_link_libraries(${PROJECT_NAME} PRIVATE parmetis MPI::MPI_Fortran)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wno-unused-function>)
endif ()

if (EZP_ENABLE_OPENMP_MUMPS)
    find_package(OpenMP REQUIRED COMPONENTS C Fortran)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_Fortran)
endif ()
