#pragma once
// How a row reads the meta edges of its canvas (include/table_abi.h "Meta edges listen to their member edges").
// A row has no graph handle, and a plugin DLL carries its own copy of the canvas statics, so it asks nodus_headless
// (the module that owns the canvas of this process) through the two process-wide readers of
// include/nodus_headless_abi.h, found by name in the already-loaded nodus_headless (nodus_headless.dll on Windows; the
// process's global symbols elsewhere, where libnodus_headless.so is loaded). Inside the image they resolve to
// the image's own functions; with no nodus_headless loaded (a stand-alone test) they report nothing.
#include <cstdint>

#if defined(_WIN32)
#include <winsock2.h>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nodus::meta {

inline void* headless_symbol(const char* name) {
#if defined(_WIN32)
    HMODULE module = GetModuleHandleA("nodus_headless.dll");
    return module ? reinterpret_cast<void*>(GetProcAddress(module, name)) : nullptr;
#else
    // Off Windows the headless image is already in the process (it loaded this row), so its exported symbol is found
    // in the global scope. This used to return nothing, and a row reading meta channels saw no records on Linux
    // (Oracle A1, 2026-11-21: meta_edge_headless_test's edge_introspect check).
    return dlsym(RTLD_DEFAULT, name);
#endif
}

// Latest record (GP_META_RECORD_WORDS uint64 words) of every member of every group whose channel group is `channel`.
// Returns words written.
inline int32_t channel_records(int32_t channel, uint64_t* out_words, int32_t capacity_words) {
    using Fn = int32_t (*)(int32_t, uint64_t*, int32_t);
    Fn fn = reinterpret_cast<Fn>(headless_symbol("nodus_headless_meta_channel_records"));
    return fn ? fn(channel, out_words, capacity_words) : 0;
}

// gp_table_meta_get_fifo_snapshot of the `ordinal`-th group of `channel`; floats written, 0 when there is none.
inline int32_t channel_floats(int32_t channel, int32_t ordinal, float* out, int32_t capacity) {
    using Fn = int32_t (*)(int32_t, int32_t, float*, int32_t);
    Fn fn = reinterpret_cast<Fn>(headless_symbol("nodus_headless_meta_channel_floats"));
    return fn ? fn(channel, ordinal, out, capacity) : 0;
}

}  // namespace nodus::meta
