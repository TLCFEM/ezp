/*
 *
 *  This file is part of MUMPS 5.8.0, released
 *  on Tue May  6 08:27:40 UTC 2025
 *
 *
 *  Copyright 1991-2025 CERFACS, CNRS, ENS Lyon, INP Toulouse, Inria,
 *  Mumps Technologies, University of Bordeaux.
 *
 *  This version of MUMPS is provided to you free of charge. It is
 *  released under the CeCILL-C license
 *  (see doc/CeCILL-C_V1-en.txt, doc/CeCILL-C_V1-fr.txt, and
 *  https://cecill.info/licences/Licence_CeCILL-C_V1-en.html)
 *
 */
#ifndef MUMPS_IO_BASIC_H
#define MUMPS_IO_BASIC_H
#include "mumps_compat.h"
#include "mumps_c_types.h"
#if !defined(WITHOUT_PTHREAD) && defined(MUMPS_WIN32)
#define WITHOUT_PTHREAD 1
#endif
#if defined(_AIX)
#if !defined(_ALL_SOURCE)
/* Macro needed for direct I/O on IBM AIX */
#define _ALL_SOURCE 1
#endif
#endif
#if !defined(MUMPS_WIN32)
#if !defined(_XOPEN_SOURCE)
/* Setting this macro avoids the warnings ("missing
 * prototype") related to the use of pread /pwrite */
#define _XOPEN_SOURCE 500
#endif
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if !defined(MUMPS_WIN32)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <time.h>
#endif
#if (defined(sgi) || defined(__sgi)) || defined(_AIX) || (defined(sun) || defined(__sun)) || defined(_GNU_SOURCE)
#undef WITH_PFUNC
#define WITH_PFUNC
#endif
#define IO_SYNC 0
#define IO_ASYNC_TH 1
#define IO_ASYNC_AIO 2
#define IO_READ 1
#define IO_WRITE 0
#define UNINITIALIZED "NAME_NOT_INITIALIZED"
#define MUMPS_OOC_DEFAULT_DIR "/tmp"
#if defined(MUMPS_WIN32)
#define SEPARATOR "\\"
#else
#define SEPARATOR "/"
#endif
/* #define NB_FILE_TYPE_FACTO 1 */
/* #define NB_FILE_TYPE_SOLVE 1 */
#define my_max(x, y) ((x) > (y) ? (x) : (y))
#define my_ceil(x) ((MUMPS_INT8)(x) >= (x) ? (MUMPS_INT8)(x) : ((MUMPS_INT8)(x) + 1))
typedef struct __mumps_file_struct {
    MUMPS_OFF_T write_pos;
    MUMPS_OFF_T current_pos;
    MUMPS_INT is_opened;
#if !defined(MUMPS_WIN32)
    MUMPS_INT file;
#else
    FILE* file;
#endif
    char name[1300]; /* Larger than prefix(255)+tmpdir(1023)+base_name (20)+\0 (1) */
} mumps_file_struct;
typedef struct __mumps_file_type {
#if !defined(MUMPS_WIN32)
    MUMPS_INT mumps_flag_open;
#else
    char mumps_flag_open[6];
#endif
    MUMPS_INT mumps_io_current_file_number;
    MUMPS_INT mumps_io_last_file_opened;
    MUMPS_INT mumps_io_nb_file_opened;
    MUMPS_INT mumps_io_nb_file;
    mumps_file_struct* mumps_io_pfile_pointer_array;
    mumps_file_struct* mumps_io_current_file;
} mumps_file_type;
/* Exported global variables */
#if !defined(MUMPS_WIN32)
#if defined(WITH_PFUNC) && !defined(WITHOUT_PTHREAD)
#include <pthread.h>
extern pthread_mutex_t mumps_io_pwrite_mutex;
#endif
#endif /* MUMPS_WIN32 */
extern mumps_file_type* mumps_files;
extern char* mumps_ooc_file_prefix;
extern MUMPS_INT mumps_elementary_data_size;
extern MUMPS_INT mumps_io_is_init_called;
extern MUMPS_INT mumps_io_myid;
extern MUMPS_OFF_T mumps_io_max_file_size;
extern MUMPS_INT mumps_io_flag_async;
extern MUMPS_INT mumps_io_k211;
extern MUMPS_INT mumps_io_nb_file_type;
/* Exported functions */
MUMPS_INT mumps_set_file(MUMPS_INT type, MUMPS_INT file_number_arg);
void mumps_update_current_file_position(mumps_file_struct* file_arg);
MUMPS_INT mumps_compute_where_to_write(const double to_be_written, const MUMPS_INT type, long long vaddr, size_t already_written);
MUMPS_INT mumps_prepare_pointers_for_write(double to_be_written, MUMPS_OFF_T* pos_in_file, MUMPS_INT* file_number, const MUMPS_INT type, long long vaddr, size_t already_written);
MUMPS_INT mumps_io_do_write_block(void* address_block, long long block_size, MUMPS_INT* type, long long vaddr, MUMPS_INT* ierr);
MUMPS_INT mumps_io_do_read_block(void* address_block, long long block_size, MUMPS_INT* type, long long vaddr, MUMPS_INT* ierr);
MUMPS_INT mumps_compute_nb_concerned_files(long long block_size, MUMPS_INT* nb_concerned_files, long long vaddr);
MUMPS_INT mumps_free_file_pointers(MUMPS_INT* step);
MUMPS_INT mumps_init_file_structure(MUMPS_INT* _myid, long long* total_size_io, MUMPS_INT* size_element, MUMPS_INT* nb_file_type, MUMPS_INT* flag_tab, MUMPS_INT keep255);
MUMPS_INT mumps_init_file_name(char* mumps_dir, char* mumps_file, MUMPS_INT* mumps_dim_dir, MUMPS_INT* mumps_dim_file, MUMPS_INT* _myid);
void mumps_io_init_file_struct(MUMPS_INT* nb, MUMPS_INT which);
MUMPS_INT mumps_io_alloc_file_struct(MUMPS_INT* nb, MUMPS_INT which);
MUMPS_INT mumps_io_get_nb_files(MUMPS_INT* nb_files, const MUMPS_INT* type);
MUMPS_INT mumps_io_get_file_name(MUMPS_INT* indice, char* name, MUMPS_INT* length, MUMPS_INT* type);
MUMPS_INT mumps_io_alloc_pointers(MUMPS_INT* nb_file_type, MUMPS_INT* dim);
MUMPS_INT mumps_io_init_vars(MUMPS_INT* myid_arg, MUMPS_INT* size_element, MUMPS_INT* async_arg, MUMPS_INT keep255);
MUMPS_INT mumps_io_set_file_name(MUMPS_INT* indice, char* name, MUMPS_INT* length, MUMPS_INT* type);
MUMPS_INT mumps_io_open_files_for_read();
MUMPS_INT mumps_io_set_last_file(MUMPS_INT* dim, MUMPS_INT* type);
MUMPS_INT mumps_io_write__(void* file, void* loc_add, size_t write_size, MUMPS_OFF_T where, MUMPS_INT type);
#if !defined(MUMPS_WIN32)
MUMPS_INT mumps_io_write_os_buff__(void* file, void* loc_add, size_t write_size, MUMPS_OFF_T where);
MUMPS_INT mumps_io_flush_write__(MUMPS_INT type);
#else
MUMPS_INT mumps_io_write_win32__(void* file, void* loc_add, size_t write_size, MUMPS_OFF_T where);
#endif
MUMPS_OFF_T mumps_io_read__(void* file, void* loc_addr, size_t size, MUMPS_OFF_T local_offset, MUMPS_INT type);
#if !defined(MUMPS_WIN32)
MUMPS_OFF_T mumps_io_read_os_buff__(void* file, void* loc_addr, size_t size, MUMPS_OFF_T local_offset);
#else
MUMPS_OFF_T mumps_io_read_win32__(void* file, void* loc_addr, size_t size, MUMPS_OFF_T local_offset);
#endif
#if !defined(MUMPS_WIN32) && !defined(WITHOUT_PTHREAD)
#if defined(WITH_PFUNC)
MUMPS_INT mumps_io_protect_pointers();
MUMPS_INT mumps_io_unprotect_pointers();
MUMPS_INT mumps_io_init_pointers_lock();
MUMPS_INT mumps_io_destroy_pointers_lock();
#endif /* WITH_PFUNC */
#endif /* MUMPS_WIN32 */
#endif /* MUMPS_IO_BASIC_H */
