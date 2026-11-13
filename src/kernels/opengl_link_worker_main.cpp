#include "opengl_protected_protocol.h"
#include "mem_backend.h"
#include <cstdio>
#include <cwchar>
int wmain(int argc,wchar_t** argv) {
    // CPU-only linkage check. No context or get-proc callback is installed and
    // no shader helper is invoked. This checks the exact registry import used
    // by this executable's native helper, independently of test mock symbols.
    if(argc==2 && std::wcscmp(argv[1],L"--verify-registry-only")==0) {
        constexpr uintptr_t id=0x4e505247u,marker=0x43505531u;
        if(gp_mem_backend_get_registered_gl_context(id))return 82;
        if(!gp_mem_backend_register_gl_context(id,marker))return 83;
        const bool matched=gp_mem_backend_get_registered_gl_context(id)==marker;
        const bool removed=gp_mem_backend_unregister_gl_context(id)!=0;
        const bool absent=gp_mem_backend_get_registered_gl_context(id)==0;
        if(!matched || !removed || !absent)return 84;
        std::puts("PASS: native worker single-registry registration/lookup/removal; no GL context or shader calls");
        return 0;
    }
    return nodus::gpu::protected_detail::protected_child_main(
        argc,argv,nodus::gpu::protected_detail::link_in_affinity_context);
}
