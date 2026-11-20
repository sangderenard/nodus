// Native child-only context owner. Unsupported affinity/identity is a refusal,
// never a default-adapter link. No dispatch, framebuffer, draw or readback API.
#include "opengl_protected_protocol.h"
#include "mem_backend.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// ===== PLATFORM REGION (the only OS-specific code in this file) =====
// Provides, for the shared code below: platform::procedure / current_context
// (the gp_gl_compute_context callbacks), GL entry wrappers, the device-identity
// reader, and platform::ContextOwner (open/close of the exact OpenGL 4.3 core
// context).
// Windows: WGL bootstrap window, WGL_NV_gpu_affinity, GL_EXT_memory_object_win32
// device LUID.
// POSIX: EGL loaded with dlopen (libEGL.so.1 / libOpenGL.so.0 or libGL.so.1) with
// the few types and entry points declared locally, so no EGL/GL headers or link
// libraries are needed. Headless: EGL_MESA_platform_surfaceless, or a device
// display picked by EGL_EXT_device_persistent_id, with a 1x1 pbuffer if the
// display lacks EGL_KHR_surfaceless_context. The 8-byte "adapter LUID" of the
// protocol is, on POSIX, the first 8 bytes of GL_DEVICE_UUID_EXT (Linux has no
// LUID); the node mask is the constant 1 (a GL context on Linux has no node mask).
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GL/gl.h>
namespace nodus::gpu::protected_detail::platform {
using GpuHandle=void*;
struct GpuDevice {DWORD cb;CHAR name[32],description[128];DWORD flags;RECT virtual_screen;};
using EnumGpus=BOOL(WINAPI*)(UINT,GpuHandle*);
using EnumDevices=BOOL(WINAPI*)(GpuHandle,UINT,GpuDevice*);
using CreateAffinity=HDC(WINAPI*)(const GpuHandle*);
using DeleteAffinity=BOOL(WINAPI*)(HDC);
using CreateContext=HGLRC(WINAPI*)(HDC,HGLRC,const int*);
using GetStringi=const GLubyte*(APIENTRY*)(GLenum,GLuint);
using GetUnsignedBytes=void(APIENTRY*)(GLenum,GLubyte*);
constexpr GLenum gl_device_luid=0x9599,gl_device_node_mask=0x959a;
constexpr const char* identity_extension="GL_EXT_memory_object_win32";

void* procedure(const char* name,void*) {
    const auto address=wglGetProcAddress(name);const auto value=reinterpret_cast<uintptr_t>(address);
    if(address && value!=1 && value!=2 && value!=3 && value!=~uintptr_t(0))return reinterpret_cast<void*>(address);
    return reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"opengl32.dll"),name));
}
uintptr_t current_context(void*) {return reinterpret_cast<uintptr_t>(wglGetCurrentContext());}
GetStringi get_stringi(){return reinterpret_cast<GetStringi>(procedure("glGetStringi",nullptr));}
void gl_get_integerv(GLenum name,GLint* value){glGetIntegerv(name,value);}
GLenum gl_get_error(){return glGetError();}
bool identity_reader_available(){return procedure("glGetUnsignedBytevEXT",nullptr)!=nullptr;}
// Writes the GL device LUID and node mask of the current context.
void read_identity(std::array<uint8_t,8>& luid,GLint& node_mask) {
    const auto get_bytes=reinterpret_cast<GetUnsignedBytes>(procedure("glGetUnsignedBytevEXT",nullptr));
    get_bytes(gl_device_luid,luid.data());glGetIntegerv(gl_device_node_mask,&node_mask);
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
    uintptr_t id()const{return reinterpret_cast<uintptr_t>(context);}
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
#else
#include <dlfcn.h>
typedef unsigned int GLenum;
typedef int GLint;
typedef unsigned int GLuint;
typedef unsigned char GLubyte;
namespace nodus::gpu::protected_detail::platform {
using GetStringi=const GLubyte*(*)(GLenum,GLuint);
using GetUnsignedBytes=void(*)(GLenum,GLubyte*);
// Device UUID query of GL_EXT_memory_object (the Linux identity source).
constexpr GLenum gl_device_uuid=0x9597;
constexpr const char* identity_extension="GL_EXT_memory_object";

// ---- minimal EGL 1.5 / EGL_EXT declarations (no system headers) ----
using EGLDisplay=void*;using EGLConfig=void*;using EGLContext=void*;using EGLSurface=void*;using EGLDeviceEXT=void*;
using EGLint=int32_t;using EGLBoolean=unsigned int;using EGLenum=unsigned int;
constexpr EGLint egl_none=0x3038,egl_extensions=0x3055,egl_surface_type=0x3033,egl_pbuffer_bit=0x0001,
    egl_renderable_type=0x3040,egl_opengl_bit=0x0008,egl_width=0x3057,egl_height=0x3056,
    egl_context_major=0x3098,egl_context_minor=0x30fb,egl_context_profile=0x30fd,egl_core_profile_bit=0x1,
    egl_device_uuid=0x335c;
constexpr EGLenum egl_opengl_api=0x30a2,egl_platform_device=0x313f,egl_platform_surfaceless=0x31dd;

struct EglApi {
    void* egl_lib=nullptr;void* gl_lib=nullptr;bool loaded=false,attempted=false;
    void* (*get_proc)(const char*)=nullptr;
    EGLDisplay (*get_display)(void*)=nullptr;
    EGLBoolean (*initialize)(EGLDisplay,EGLint*,EGLint*)=nullptr;
    EGLBoolean (*terminate)(EGLDisplay)=nullptr;
    const char* (*query_string)(EGLDisplay,EGLint)=nullptr;
    EGLBoolean (*bind_api)(EGLenum)=nullptr;
    EGLBoolean (*choose_config)(EGLDisplay,const EGLint*,EGLConfig*,EGLint,EGLint*)=nullptr;
    EGLContext (*create_context)(EGLDisplay,EGLConfig,EGLContext,const EGLint*)=nullptr;
    EGLBoolean (*destroy_context)(EGLDisplay,EGLContext)=nullptr;
    EGLSurface (*create_pbuffer)(EGLDisplay,EGLConfig,const EGLint*)=nullptr;
    EGLBoolean (*destroy_surface)(EGLDisplay,EGLSurface)=nullptr;
    EGLBoolean (*make_current)(EGLDisplay,EGLSurface,EGLSurface,EGLContext)=nullptr;
    EGLContext (*get_current_context)()=nullptr;
    // Never unloaded: the driver may keep threads and handlers alive.
    bool load() {
        if(attempted)return loaded;
        attempted=true;
        egl_lib=dlopen("libEGL.so.1",RTLD_NOW|RTLD_LOCAL);
        if(!egl_lib)return false;
        gl_lib=dlopen("libOpenGL.so.0",RTLD_NOW|RTLD_LOCAL);
        if(!gl_lib)gl_lib=dlopen("libGL.so.1",RTLD_NOW|RTLD_LOCAL);
        const auto sym=[&](const char* name){return dlsym(egl_lib,name);};
#define NODUS_EGL(field,name) *reinterpret_cast<void**>(&field)=sym(name)
        NODUS_EGL(get_proc,"eglGetProcAddress");NODUS_EGL(get_display,"eglGetDisplay");
        NODUS_EGL(initialize,"eglInitialize");NODUS_EGL(terminate,"eglTerminate");
        NODUS_EGL(query_string,"eglQueryString");NODUS_EGL(bind_api,"eglBindAPI");
        NODUS_EGL(choose_config,"eglChooseConfig");NODUS_EGL(create_context,"eglCreateContext");
        NODUS_EGL(destroy_context,"eglDestroyContext");NODUS_EGL(create_pbuffer,"eglCreatePbufferSurface");
        NODUS_EGL(destroy_surface,"eglDestroySurface");NODUS_EGL(make_current,"eglMakeCurrent");
        NODUS_EGL(get_current_context,"eglGetCurrentContext");
#undef NODUS_EGL
        loaded=get_proc && get_display && initialize && terminate && query_string && bind_api && choose_config &&
               create_context && destroy_context && create_pbuffer && destroy_surface && make_current && get_current_context;
        return loaded;
    }
};
EglApi& egl() {static EglApi api;return api;}
bool has_token(const char* list,const char* token) {
    if(!list)return false;
    const size_t n=std::strlen(token);
    for(const char* p=list;*p;) {
        while(*p==' ')++p;
        const char* end=p;while(*end && *end!=' ')++end;
        if(static_cast<size_t>(end-p)==n && !std::strncmp(p,token,n))return true;
        p=end;
    }
    return false;
}

void* procedure(const char* name,void*) {
    auto& api=egl();if(!api.load())return nullptr;
    if(void* address=api.get_proc(name))return address;
    return api.gl_lib?dlsym(api.gl_lib,name):nullptr;
}
uintptr_t current_context(void*) {
    auto& api=egl();return api.load()?reinterpret_cast<uintptr_t>(api.get_current_context()):0;
}
GetStringi get_stringi(){return reinterpret_cast<GetStringi>(procedure("glGetStringi",nullptr));}
void gl_get_integerv(GLenum name,GLint* value) {
    using Fn=void(*)(GLenum,GLint*);
    const auto fn=reinterpret_cast<Fn>(procedure("glGetIntegerv",nullptr));
    if(fn)fn(name,value);
}
GLenum gl_get_error() {
    using Fn=GLenum(*)();
    const auto fn=reinterpret_cast<Fn>(procedure("glGetError",nullptr));
    return fn?fn():0x0502; // GL_INVALID_OPERATION: an unloadable error query is not "no error"
}
bool identity_reader_available(){return procedure("glGetUnsignedBytevEXT",nullptr)!=nullptr;}
// Writes the device identity of the current context: first 8 bytes of
// GL_DEVICE_UUID_EXT, and node mask 1.
void read_identity(std::array<uint8_t,8>& luid,GLint& node_mask) {
    const auto get_bytes=reinterpret_cast<GetUnsignedBytes>(procedure("glGetUnsignedBytevEXT",nullptr));
    GLubyte uuid[16]{};get_bytes(gl_device_uuid,uuid);
    std::memcpy(luid.data(),uuid,luid.size());node_mask=1;
}

struct ContextOwner {
    EGLDisplay display=nullptr;EGLContext context=nullptr;EGLSurface surface=nullptr;
    uintptr_t registered=0;
    uintptr_t id()const{return reinterpret_cast<uintptr_t>(context);}
    bool close() {
        bool ok=true;auto& api=egl();
        if(display && context && api.get_current_context()==context)
            if(!api.make_current(display,nullptr,nullptr,nullptr))ok=false;
        if(registered){if(!gp_mem_backend_unregister_gl_context(registered))ok=false;registered=0;}
        if(context){if(!api.destroy_context(display,context))ok=false;context=nullptr;}
        if(surface){if(!api.destroy_surface(display,surface))ok=false;surface=nullptr;}
        if(display){if(!api.terminate(display))ok=false;display=nullptr;}
        return ok;
    }
    ~ContextOwner(){close();}
    // OpenGL 4.3 core on an initialized display; no fallback to an older context.
    bool make_context(EGLDisplay candidate,std::string& error,const char* failure) {
        auto& api=egl();EGLint major=0,minor=0;
        if(!api.initialize(candidate,&major,&minor)){error="EGL display initialization failed";return false;}
        display=candidate;
        if(!api.bind_api(egl_opengl_api)){error="EGL cannot bind desktop OpenGL";return false;}
        const EGLint config_attributes[]{egl_surface_type,egl_pbuffer_bit,egl_renderable_type,egl_opengl_bit,egl_none};
        EGLConfig config=nullptr;EGLint count=0;
        if(!api.choose_config(display,config_attributes,&config,1,&count) || count<1){error="EGL has no desktop OpenGL configuration";return false;}
        const EGLint attributes[]{egl_context_major,4,egl_context_minor,3,egl_context_profile,egl_core_profile_bit,egl_none};
        context=api.create_context(display,config,nullptr,attributes);
        if(!context){error=failure;return false;}
        // Prefer no surface at all; a 1x1 pbuffer only where the display needs one.
        if(!has_token(api.query_string(display,egl_extensions),"EGL_KHR_surfaceless_context")) {
            const EGLint size[]{egl_width,1,egl_height,1,egl_none};
            surface=api.create_pbuffer(display,config,size);
            if(!surface){error="EGL pbuffer surface creation failed";return false;}
        }
        if(!api.make_current(display,surface,surface,context)){error=failure;return false;}
        return true;
    }
    bool open(const ChildRequest& request,std::string& error) {
        auto& api=egl();
        if(!api.load()){error="Child EGL library (libEGL.so.1) could not be loaded";return false;}
        using GetPlatformDisplay=EGLDisplay(*)(EGLenum,void*,const EGLint*);
        const auto get_platform_display=reinterpret_cast<GetPlatformDisplay>(api.get_proc("eglGetPlatformDisplayEXT"));
        const char* client=api.query_string(nullptr,egl_extensions);
        if(request.context_policy==ProtectedContextPolicy::VerifiedDefaultDc) {
            // This policy does not claim exact-device selection or try other GPUs.
            // The caller below MUST prove the actual device identity before any
            // shader call. A mismatch returns immediately, without retry.
            EGLDisplay candidate=nullptr;
            if(get_platform_display && has_token(client,"EGL_MESA_platform_surfaceless"))
                candidate=get_platform_display(egl_platform_surfaceless,nullptr,nullptr);
            if(candidate && make_context(candidate,error,"Default-display OpenGL 4.3 context creation failed"))return true;
            if(candidate){close();error.clear();}
            candidate=api.get_display(nullptr);
            if(!candidate){error="EGL has no default display";return false;}
            return make_context(candidate,error,"Default-display OpenGL 4.3 context creation failed");
        }
        // Exact selection: the EGL device whose persistent UUID matches the request.
        using QueryDevices=EGLBoolean(*)(EGLint,EGLDeviceEXT*,EGLint*);
        using QueryBinary=EGLBoolean(*)(EGLDeviceEXT,EGLint,EGLint,void*,EGLint*);
        const auto query_devices=reinterpret_cast<QueryDevices>(api.get_proc("eglQueryDevicesEXT"));
        const auto query_binary=reinterpret_cast<QueryBinary>(api.get_proc("eglQueryDeviceBinaryEXT"));
        if(!get_platform_display || !query_devices || !query_binary) {
            error="Verified EGL_EXT_device_persistent_id is unavailable; refusing default-adapter shader link";return false;
        }
        EGLDeviceEXT devices[64]{};EGLint count=0;
        if(!query_devices(64,devices,&count) || count<0){error="EGL device enumeration failed";return false;}
        EGLDeviceEXT selected=nullptr;
        for(EGLint i=0;i<count && i<64;++i) {
            uint8_t uuid[16]{};EGLint size=0;
            if(!query_binary(devices[i],egl_device_uuid,sizeof(uuid),uuid,&size) || size!=16)continue;
            if(std::memcmp(uuid,request.adapter_luid.data(),request.adapter_luid.size())==0) {
                if(selected && selected!=devices[i]){error="Requested adapter identity maps ambiguously to EGL devices";return false;}
                selected=devices[i];
            }
        }
        if(!selected){error="Requested adapter identity has no verified EGL device mapping";return false;}
        const EGLDisplay candidate=get_platform_display(egl_platform_device,selected,nullptr);
        if(!candidate){error="EGL refused the exact requested device";return false;}
        return make_context(candidate,error,"Exact-device OpenGL 4.3 context creation failed");
    }
};
}
#endif
// ===== END PLATFORM REGION =====

namespace nodus::gpu::protected_detail {
namespace {
using platform::ContextOwner;
constexpr GLenum gl_extensions=0x1f03,gl_num_extensions=0x821d;
constexpr GLenum gl_no_error=0;
bool has_extension(const char* name) {
    const auto stringi=platform::get_stringi();
    if(!stringi)return false;
    GLint count=0;platform::gl_get_integerv(gl_num_extensions,&count);
    if(platform::gl_get_error()!=gl_no_error || count<0 || count>65536)return false;
    for(GLint i=0;i<count;++i) {
        const auto* extension=stringi(gl_extensions,static_cast<GLuint>(i));
        if(extension && std::strcmp(reinterpret_cast<const char*>(extension),name)==0)return true;
    }
    return false;
}
}

ProtectedLinkOutcome link_in_affinity_context(const ChildRequest& request) {
    ProtectedLinkOutcome out;out.host_id=request.link.host_id;out.contract_id=request.link.contract_id;
    out.generation=request.link.generation;out.requested_luid=request.adapter_luid;
    out.status=ProtectedLinkStatus::IdentityUnavailable;
    ContextOwner owner;
    if(!owner.open(request,out.message)){out.resources_released=owner.close();return out;}
    out.context_selection=request.context_policy==ProtectedContextPolicy::VerifiedDefaultDc?
        ProtectedContextSelection::VerifiedDefaultDc:ProtectedContextSelection::NvAffinity;
    GLint node_mask=0;
    if(!has_extension(platform::identity_extension) || !platform::identity_reader_available()) {
        out.message="Created context cannot expose a verifiable GL device LUID; no shader call";
        out.resources_released=owner.close();return out;
    }
    platform::read_identity(out.observed_luid,node_mask);
    if(platform::gl_get_error()!=gl_no_error || out.observed_luid!=request.adapter_luid ||
       static_cast<uint32_t>(node_mask)!=request.node_mask) {
        out.message="Actual GL device LUID/node mask differs from the requested GPU; no shader call";
        out.resources_released=owner.close();return out;
    }
    out.identity_verified=true;
    const auto id=owner.id();
    if(!gp_mem_backend_register_gl_context(id,id)) {
        out.message="Native memory backend refused exact child-owned GL context registration";
        out.resources_released=owner.close();return out;
    }
    owner.registered=id;
    gp_gl_compute_context context{id,platform::procedure,platform::current_context,nullptr};gp_gl_compute_program* program=nullptr;
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
}
} // namespace nodus::gpu::protected_detail
