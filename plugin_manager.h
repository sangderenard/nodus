#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Load a plugin DLL from path. Returns non-zero on success and fills returned id into out_id (must be non-null).
int gp_plugin_load_from_path(const char* path, char* out_id, int out_id_capacity);
// same as above but passes a HostAPI pointer (cast as void*) to the plugin's init function
int gp_plugin_load_from_path_with_host(const char* path, void* host, char* out_id, int out_id_capacity);

// Unload a previously loaded plugin by id. Returns non-zero on success.
int gp_plugin_unload(const char* id);

// Get number of loaded plugins (for iteration) and optionally fill a buffer of ids (comma separated) if provided.
int gp_plugin_list_ids(char* out_buf, int out_buf_capacity);

// Build a CMake target and load the resulting DLL.
// Parameters:
//  - build_dir: path to CMake build directory
//  - target: CMake target name to build
//  - built_relpath: relative path from build_dir to the produced DLL (e.g., "Release\\my_module.dll")
//  - dest_dir: destination folder to copy a timestamped DLL into (may be null to use current dir)
//  - out_id: output buffer to receive registered tool id
//  - out_id_capacity: capacity of out_id buffer
// Returns 1 on success, 0 on failure.
int gp_plugin_build_and_load(const char* build_dir,
							 const char* target,
							 const char* built_relpath,
							 const char* dest_dir,
							 char* out_id,
							 int out_id_capacity);

int gp_plugin_build_and_load_with_host(const char* build_dir,
									   const char* target,
									   const char* built_relpath,
									   const char* dest_dir,
									   void* host,
									   char* out_id,
									   int out_id_capacity);

// Build a single source file in a scratch build directory and load the produced shared
// library. `module_src` should be an absolute path to the module source (.cpp).
// `repo_root` is used as an include directory for the scratch CMake project and may
// be null. Returns 1 on success and writes the registered id to out_id.
int gp_plugin_build_module_and_load(const char* module_src,
									const char* repo_root,
									const char* dest_dir,
									void* host,
									char* out_id,
									int out_id_capacity);

#ifdef __cplusplus
}
#endif
