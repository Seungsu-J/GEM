#ifndef JOIN_HOST_SELECT_CUH
#define JOIN_HOST_SELECT_CUH

#include "join_helpers.h"

void host_select();

void direct_select();
void direct_select_debug();

void refer_select();
void refer_select_debug();

void direct_select_write_pipeline();
void direct_select_pipeline();
void refer_select_write_pipeline();
void refer_select_pipeline();

#endif
