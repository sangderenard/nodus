#pragma once

#include "module_library.h"

int gp_module_library_actualize_sources(const GP_ModuleLibrary& library, const char* output_root);
int gp_module_library_actualize_from_file(const char* path, const char* output_root);
