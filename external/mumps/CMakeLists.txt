cmake_minimum_required(VERSION 3.18)

project(mumps C Fortran)

add_compile_definitions(Add_)
add_compile_definitions(pord)
# add_compile_definitions(metis)

if (MKL_FOUND)
    add_compile_definitions(GEMMT_AVAILABLE)
endif ()

if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -fopenmp -w -fallow-argument-mismatch -fno-strict-aliasing --param=max-vartrack-size=500000000")
elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp -qopenmp")
endif ()

file(GLOB mumps_f "*.F")
file(GLOB mumps_c "*.c")

set(mumps_mod
        mumps_pivnul_mod.F
        ana_omp_m.F
        ana_blk_m.F
        ana_orderings_wrappers_m.F
        double_linked_list.F
        fac_asm_build_sort_index_ELT_m.F
        fac_asm_build_sort_index_m.F
        fac_descband_data_m.F
        fac_future_niv2_mod.F
        fac_ibct_data_m.F
        fac_maprow_data_m.F
        front_data_mgt_m.F
        lr_common.F
        mumps_comm_ibcast.F
        mumps_l0_omp_m.F
        mumps_memory_mod.F
        mumps_mpitoomp_m.F
        mumps_ooc_common.F
        mumps_static_mapping.F
        omp_tps_common_m.F

        dana_aux.F
        dana_aux_par.F
        dana_lr.F
        dfac_mem_dynamic.F
        dfac_asm_master_ELT_m.F
        dfac_asm_master_m.F
        dfac_front_aux.F
        dfac_front_LU_type1.F
        dfac_front_LU_type2.F
        dfac_front_LDLT_type1.F
        dfac_front_LDLT_type2.F
        dfac_front_type2_aux.F
        dfac_sispointers_m.F
        dfac_lr.F
        dfac_omp_m.F
        dfac_par_m.F
        dfac_sol_l0omp_m.F
        dlr_core.F
        dlr_stats.F
        dlr_type.F
        dmumps_comm_buffer.F
        dmumps_config_file.F
        dmumps_load.F
        dmumps_lr_data_m.F
        dmumps_ooc_buffer.F
        dmumps_ooc.F
        dmumps_sol_es.F
        dmumps_save_restore.F
        dmumps_save_restore_files.F
        dmumps_struc_def.F
        domp_tps_m.F
        dsol_lr.F
        dsol_omp_m.F
        dstatic_ptr_m.F
        dmumps_mpi3_mod.F

        sana_aux.F
        sana_aux_par.F
        sana_lr.F
        sfac_mem_dynamic.F
        sfac_asm_master_ELT_m.F
        sfac_asm_master_m.F
        sfac_front_aux.F
        sfac_front_LU_type1.F
        sfac_front_LU_type2.F
        sfac_front_LDLT_type1.F
        sfac_front_LDLT_type2.F
        sfac_front_type2_aux.F
        sfac_sispointers_m.F
        sfac_lr.F
        sfac_omp_m.F
        sfac_par_m.F
        sfac_sol_l0omp_m.F
        slr_core.F
        slr_stats.F
        slr_type.F
        smumps_comm_buffer.F
        smumps_config_file.F
        smumps_load.F
        smumps_lr_data_m.F
        smumps_ooc_buffer.F
        smumps_ooc.F
        smumps_sol_es.F
        smumps_save_restore.F
        smumps_save_restore_files.F
        smumps_struc_def.F
        somp_tps_m.F
        ssol_lr.F
        ssol_omp_m.F
        sstatic_ptr_m.F
        smumps_mpi3_mod.F

        zana_aux.F
        zana_aux_par.F
        zana_lr.F
        zfac_mem_dynamic.F
        zfac_asm_master_ELT_m.F
        zfac_asm_master_m.F
        zfac_front_aux.F
        zfac_front_LU_type1.F
        zfac_front_LU_type2.F
        zfac_front_LDLT_type1.F
        zfac_front_LDLT_type2.F
        zfac_front_type2_aux.F
        zfac_sispointers_m.F
        zfac_lr.F
        zfac_omp_m.F
        zfac_par_m.F
        zfac_sol_l0omp_m.F
        zlr_core.F
        zlr_stats.F
        zlr_type.F
        zmumps_comm_buffer.F
        zmumps_config_file.F
        zmumps_load.F
        zmumps_lr_data_m.F
        zmumps_ooc_buffer.F
        zmumps_ooc.F
        zmumps_sol_es.F
        zmumps_save_restore.F
        zmumps_save_restore_files.F
        zmumps_struc_def.F
        zomp_tps_m.F
        zsol_lr.F
        zsol_omp_m.F
        zstatic_ptr_m.F
        zmumps_mpi3_mod.F

        cana_aux.F
        cana_aux_par.F
        cana_lr.F
        cfac_mem_dynamic.F
        cfac_asm_master_ELT_m.F
        cfac_asm_master_m.F
        cfac_front_aux.F
        cfac_front_LU_type1.F
        cfac_front_LU_type2.F
        cfac_front_LDLT_type1.F
        cfac_front_LDLT_type2.F
        cfac_front_type2_aux.F
        cfac_sispointers_m.F
        cfac_lr.F
        cfac_omp_m.F
        cfac_par_m.F
        cfac_sol_l0omp_m.F
        clr_core.F
        clr_stats.F
        clr_type.F
        cmumps_comm_buffer.F
        cmumps_config_file.F
        cmumps_load.F
        cmumps_lr_data_m.F
        cmumps_ooc_buffer.F
        cmumps_ooc.F
        cmumps_sol_es.F
        cmumps_save_restore.F
        cmumps_save_restore_files.F
        cmumps_struc_def.F
        comp_tps_m.F
        csol_lr.F
        csol_omp_m.F
        cstatic_ptr_m.F
        cmumps_mpi3_mod.F
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

message(STATUS "MUMPS Fortran Flags: ${CMAKE_Fortran_FLAGS}")

add_library(${PROJECT_NAME} ${mumps_c} ${mumps_f} ${mumps_mod})

target_link_libraries(${PROJECT_NAME} gfortran gomp ${MPI_FORT_LIB})
