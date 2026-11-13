// Native child-only context owner. Unsupported affinity/identity is a refusal,
// never a default-adapter link. No dispatch, framebuffer, draw or readback API.
#include "opengl_protected_protocol.h"
#include "mem_backend.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GL/gl.h>
#endif

namespace nodus::gpu::protected_detail {
#ifdef _WIN32
namespace {
using GpuHandle=void*;
struct GpuDevice {DWORD cb;CHAR name[32],description[128];DWORD flags;RECT virtual_screen;};
using EnumGpus=BOOL(WINAPI*)(UINT,GpuHandle*);
using EnumDevices=BOOL(WINAPI*)(GpuHandle,UINT,GpuDevice*);
using CreateAffinity=HDC(WINAPI*)(const GpuHandle*);
using DeleteAffinity=BOOL(WINAPI*)(HDC);
using CreateContext=HGLRC(WINAPI*)(HDC,HGLRC,const int*);
using GetStringi=const GLubyte*(APIENTRY*)(GLenum,GLuint);
using GetUnsignedBytes=void(APIENTRY*)(GLenum,GLubyte*);
constexpr GLenum gl_extensions=0x1f03,gl_num_extensions=0x821d;
constexpr GLenum gl_device_luid=0x9599,gl_device_node_mask=0x959a;

void* procedure(const char* name,void*) {
    const auto address=wglGetProcAddress(name);const auto value=reinterpret_cast<uintptr_t>(address);
    if(address && value!=1 && value!=2 && value!=3 && value!=~uintptr_t(0))return reinterpret_cast<void*>(address);
    return reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"opengl32.dll"),name));
}
uintptr_t current_context(void*) {return reinterpret_cast<uintptr_t>(wglGetCurrentContext());}
bool has_extension(const char* name) {
    const auto stringi=reinterpret_cast<GetStringi>(procedure("glGetStringi",nullptr));
    if(!stringi)return false;
    GLint count=0;glGetIntegerv(gl_num_extensions,&count);
    if(glGetError()!=GL_NO_ERROR || count<0 || count>65536)return false;
    for(GLint i=0;i<count;++i) {
        const auto* extension=stringi(gl_extensions,static_cast<GLuint>(i));
        if(extension && std::strcmp(reinterpret_cast<const char*>(extension),name)==0)return true;
    }
    return false;
}
// WGL GPU_DEVICE names are GDI display names. Match them to the exact Windows
// display-source adapter LUID, not WMI order, vendor strings or Vulkan ordinal.
bool display_luid(const char* name,std::array<uint8_t,8>& luid) {
    wchar_t wide[32]{};
    if(!MultiByteToWideChar(CP_ACP,MB_ERR_INVALID_CHARS,name,-1,wide,32))return false;
    UINT32 path_count=0,mode_count=0;
    if(GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS,&path_count,&mode_count)!=ERROR_SUCCESS ||
       path_count>1024 || mode_count>4096)return false;
    std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
    std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
    if(QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS,&path_count,paths.data(),&mode_count,modes.data(),nullptr)!=ERROR_SUCCESS)return false;
    bool found=false;
    for(UINT32 i=0;i<path_count;++i) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};source.header.type=DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source.header.size=sizeof(source);source.header.adapterId=paths[i].sourceInfo.adapterId;
        source.header.id=paths[i].sourceInfo.id;
        if(DisplayConfigGetDeviceInfo(&source.header)!=ERROR_SUCCESS || _wcsicmp(wide,source.viewGdiDeviceName))continue;
        std::array<uint8_t,8> candidate{};std::memcpy(candidate.data(),&source.header.adapterId,sizeof(LUID));
        if(found && candidate!=luid)return false;
        luid=candidate;found=true;
    }
    return found;
}
struct ContextOwner {
    HWND window=nullptr;HDC bootstrap_dc=nullptr,affinity_dc=nullptr;
    HGLRC bootstrap=nullptr,context=nullptr;DeleteAffinity delete_affinity=nullptr;
    uintptr_t registered=0;bool class_registered=false;
    static constexpr const wchar_t* window_class=L"NodusProtectedLinkBootstrapV1";
    bool close() {
        bool ok=true;
        if((context && wglGetCurrentContext()==context) || (bootstrap && wglGetCurrentContext()==bootstrap))
            if(!wglMakeCurrent(nullptr,nullptr))ok=false;
        if(registered){if(!gp_mem_backend_unregister_gl_context(registered))ok=false;registered=0;}
        if(context){if(!wglDeleteContext(context))ok=false;context=nullptr;}
        if(bootstrap){if(!wglDeleteContext(bootstrap))ok=false;bootstrap=nullptr;}
        if(affinity_dc){if(!delete_affinity || !delete_affinity(affinity_dc))ok=false;affinity_dc=nullptr;}
        if(bootstrap_dc){if(!ReleaseDC(window,bootstrap_dc))ok=false;bootstrap_dc=nullptr;}
        if(window){if(!DestroyWindow(window))ok=false;window=nullptr;}
        if(class_registered){if(!UnregisterClassW(window_class,GetModuleHandleW(nullptr)))ok=false;class_registered=false;}
        return ok;
    }
    ~ContextOwner(){close();}
    bool open(const ChildRequest& request,std::string& error) {
        WNDCLASSW cls{};cls.style=CS_OWNDC;cls.lpfnWndProc=DefWindowProcW;
        cls.hInstance=GetModuleHandleW(nullptr);cls.lpszClassName=window_class;
        if(!RegisterClassW(&cls)){error="Could not register the child-only hidden WGL bootstrap";return false;}
        class_registered=true;
        window=CreateWindowExW(0,window_class,L"",WS_POPUP,0,0,1,1,nullptr,nullptr,cls.hInstance,nullptr);
        if(!window || !(bootstrap_dc=GetDC(window))){error="Could not create hidden child WGL bootstrap DC";return false;}
        PIXELFORMATDESCRIPTOR pixel{};pixel.nSize=sizeof(pixel);pixel.nVersion=1;
        pixel.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;pixel.iPixelType=PFD_TYPE_RGBA;pixel.cColorBits=24;
        const int format=ChoosePixelFormat(bootstrap_dc,&pixel);
        if(!format || !SetPixelFormat(bootstrap_dc,format,&pixel) || !(bootstrap=wglCreateContext(bootstrap_dc)) ||
           !wglMakeCurrent(bootstrap_dc,bootstrap)){error="Child WGL bootstrap creation failed";return false;}
        const auto create_context=reinterpret_cast<CreateContext>(procedure("wglCreateContextAttribsARB",nullptr));
        if(!create_context){error="WGL cannot create the requested OpenGL 4.3 context";return false;}
        const int attributes[]{0x2091,4,0x2092,3,0x9126,1,0};
        if(request.context_policy==ProtectedContextPolicy::VerifiedDefaultDc) {
            // This policy does not claim affinity support or try other GPUs.
            // The caller below MUST prove the actual LUID and node mask before
            // any shader call. A mismatch returns immediately, without retry.
            context=create_context(bootstrap_dc,nullptr,attributes);
            if(!context || !wglMakeCurrent(bootstrap_dc,context)) {
                error="Default-DC OpenGL 4.3 context creation failed";return false;
            }
            return true;
        }
        const auto enumerate=reinterpret_cast<EnumGpus>(procedure("wglEnumGpusNV",nullptr));
        const auto devices=reinterpret_cast<EnumDevices>(procedure("wglEnumGpuDevicesNV",nullptr));
        const auto create_affinity=reinterpret_cast<CreateAffinity>(procedure("wglCreateAffinityDCNV",nullptr));
        delete_affinity=reinterpret_cast<DeleteAffinity>(procedure("wglDeleteDCNV",nullptr));
        if(!enumerate || !devices || !create_affinity || !delete_affinity) {
            error="Verified WGL_NV_gpu_affinity is unavailable; refusing default-adapter shader link";return false;
        }
        GpuHandle selected=nullptr;
        for(UINT i=0;i<64;++i) {
            GpuHandle gpu=nullptr;if(!enumerate(i,&gpu))break;
            bool matches=false;
            for(UINT j=0;j<64;++j) {
                GpuDevice device{};device.cb=sizeof(device);
                if(!devices(gpu,j,&device))break;
                if(!std::memchr(device.name,0,sizeof(device.name)))continue;
                std::array<uint8_t,8> luid{};
                if(display_luid(device.name,luid) && luid==request.adapter_luid)matches=true;
            }
            if(matches) {
                if(selected && selected!=gpu){error="Requested adapter LUID maps ambiguously to WGL GPUs";return false;}
                selected=gpu;
            }
        }
        if(!selected){error="Requested adapter LUID has no verified WGL affinity GPU/display mapping";return false;}
        const GpuHandle gpu_list[]{selected,nullptr};affinity_dc=create_affinity(gpu_list);
        if(!affinity_dc){error="wglCreateAffinityDCNV refused the exact requested GPU";return false;}
        const int affinity_format=ChoosePixelFormat(affinity_dc,&pixel);
        if(!affinity_format || !SetPixelFormat(affinity_dc,affinity_format,&pixel)) {
            error="Could not set the exact affinity DC pixel format";return false;
        }
        // OpenGL 4.3 core, no fallback to an older or default-device context.
        context=create_context(affinity_dc,nullptr,attributes);
        if(!context || !wglMakeCurrent(affinity_dc,context)){error="Exact-affinity OpenGL 4.3 context creation failed";return false;}
        return true;
    }
};
}
#endif

ProtectedLinkOutcome link_in_affinity_context(const ChildRequest& request) {
    ProtectedLinkOutcome out;out.host_id=request.link.host_id;out.contract_id=request.link.contract_id;
    out.generation=request.link.generation;out.requested_luid=request.adapter_luid;
    out.status=ProtectedLinkStatus::IdentityUnavailable;
#ifndef _WIN32
    out.message="Native protected WGL context owner is Windows-only";return out;
#else
    ContextOwner owner;
    if(!owner.open(request,out.message)){out.resources_released=owner.close();return out;}
    out.context_selection=request.context_policy==ProtectedContextPolicy::VerifiedDefaultDc?
        ProtectedContextSelection::VerifiedDefaultDc:ProtectedContextSelection::NvAffinity;
    const auto get_bytes=reinterpret_cast<GetUnsignedBytes>(procedure("glGetUnsignedBytevEXT",nullptr));
    GLint node_mask=0;
    if(!has_extension("GL_EXT_memory_object_win32") || !get_bytes) {
        out.message="Created context cannot expose a verifiable GL device LUID; no shader call";
        out.resources_released=owner.close();return out;
    }
    get_bytes(gl_device_luid,out.observed_luid.data());glGetIntegerv(gl_device_node_mask,&node_mask);
    if(glGetError()!=GL_NO_ERROR || out.observed_luid!=request.adapter_luid ||
       static_cast<uint32_t>(node_mask)!=request.node_mask) {
        out.message="Actual GL device LUID/node mask differs from the requested GPU; no shader call";
        out.resources_released=owner.close();return out;
    }
    out.identity_verified=true;
    const auto id=reinterpret_cast<uintptr_t>(owner.context);
    if(!gp_mem_backend_register_gl_context(id,id)) {
        out.message="Native memory backend refused exact child-owned GL context registration";
        out.resources_released=owner.close();return out;
    }
    owner.registered=id;
    gp_gl_compute_context context{id,procedure,current_context,nullptr};gp_gl_compute_program* program=nullptr;
    char error[2048]{};out.attempted=true;
    const bool linked=gp_gl_compute_link(&context,request.link.sources,request.link.source_count,&program,error,sizeof(error))!=0;
    out.message=error;out.status=linked?ProtectedLinkStatus::Linked:ProtectedLinkStatus::LinkFailed;
    // Failed linking may also return a residual cleanup owner. Release only in
    // this child, with this exact context current. Never return the GLuint.
    bool released=true;
    if(program) {
        char cleanup[2048]{};
        released=gp_gl_compute_release(&context,program,cleanup,sizeof(cleanup))!=0;
        if(!released){if(!out.message.empty())out.message+="; ";out.message+=cleanup;}
    }
    const bool context_closed=owner.close();out.resources_released=released && context_closed;
    out.gl_cleanup_verified=out.resources_released;
    out.linked=linked && out.resources_released;
    if(linked && !out.linked) {out.status=ProtectedLinkStatus::LinkFailed;out.message+="; explicit native GL/context cleanup was not verified";}
    return out;
#endif
}
} // namespace nodus::gpu::protected_detail
