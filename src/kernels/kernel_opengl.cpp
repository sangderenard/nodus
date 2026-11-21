#include "kernel_opengl_runner.h"
#include "mem_backend.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define GP_GL_CALL __stdcall
#else
#define GP_GL_CALL
#endif

struct gp_gl_compute_program {
    uintptr_t context_id = 0, native_context = 0;
    uint32_t name = 0;
    bool linked = false;
    std::vector<uint32_t> shaders; // retained only until checked cleanup succeeds
};

namespace {
using U = uint32_t;
using I = int32_t;
using B = uint8_t;
using P = intptr_t;
constexpr U compute_shader=0x91B9, compile_status=0x8B81, link_status=0x8B82,
    info_log_length=0x8B84, storage_buffer=0x90D2,
    max_groups=0x91BE, max_storage_bindings=0x90DD, max_image_units=0x8F38,
    storage_alignment=0x90DF, buffer_size=0x8764;

struct Api {
#define FN(ret, name, args) using name##T = ret(GP_GL_CALL*)args; name##T name = nullptr
    FN(U, CreateShader, (U));
    FN(void, ShaderSource, (U,I,const char* const*,const I*));
    FN(void, CompileShader, (U));
    FN(void, GetShaderiv, (U,U,I*));
    FN(void, GetShaderInfoLog, (U,I,I*,char*));
    FN(void, DeleteShader, (U));
    FN(U, CreateProgram, ());
    FN(void, AttachShader, (U,U));
    FN(void, DetachShader, (U,U));
    FN(void, LinkProgram, (U));
    FN(void, GetProgramiv, (U,U,I*));
    FN(void, GetProgramInfoLog, (U,I,I*,char*));
    FN(void, DeleteProgram, (U));
    FN(void, UseProgram, (U));
    FN(B, IsProgram, (U));
    FN(B, IsBuffer, (U));
    FN(B, IsTexture, (U));
    FN(U, GetError, ());
    FN(void, GetIntegerv, (U,I*));
    FN(void, GetIntegeri_v, (U,U,I*));
    FN(void, BindBuffer, (U,U));
    FN(void, GetBufferParameteri64v, (U,U,int64_t*));
    FN(void, BindBufferBase, (U,U,U));
    FN(void, BindBufferRange, (U,U,U,P,P));
    FN(void, BindImageTexture, (U,U,I,B,I,U,U));
    FN(I, GetUniformLocation, (U,const char*));
    FN(void, DispatchCompute, (U,U,U));
    FN(void, MemoryBarrier, (U));
    using IntUniform = void(GP_GL_CALL*)(I,I,const I*);
    using UIntUniform = void(GP_GL_CALL*)(I,I,const U*);
    using FloatUniform = void(GP_GL_CALL*)(I,I,const float*);
    using MatrixUniform = void(GP_GL_CALL*)(I,I,B,const float*);
    std::array<IntUniform,4> iv{};
    std::array<UIntUniform,4> uv{};
    std::array<FloatUniform,4> fv{};
    std::array<MatrixUniform,3> mv{};
#undef FN
};

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

void report(char* output, size_t capacity, std::string_view message) noexcept {
    if (!output || !capacity) return;
    const auto length=std::min(capacity-1,message.size());
    std::memcpy(output,message.data(),length); output[length]='\0';
}
uintptr_t check_context(const gp_gl_compute_context* context) {
    require(context && context->context_id && context->get_proc &&
            context->current_context,"OpenGL context callbacks/id missing");
    const auto registered=gp_mem_backend_get_registered_gl_context(context->context_id);
    require(registered && registered==context->current_context(context->user),
            "OpenGL current context does not match Nodus registered context");
    return registered;
}
template<class F> F resolve(const gp_gl_compute_context& context, const char* name) {
    const auto value=context.get_proc(name,context.user);
    const auto address=reinterpret_cast<uintptr_t>(value);
    // wglGetProcAddress also uses 1, 2, 3 and -1 for unsupported procedures.
    require(address>3 && address!=UINTPTR_MAX,std::string("OpenGL procedure unavailable: ")+name);
    return reinterpret_cast<F>(value);
}
Api load(const gp_gl_compute_context& context, bool execution) {
    Api api;
#define LOAD(name) api.name=resolve<Api::name##T>(context,"gl" #name)
    LOAD(GetError); LOAD(DeleteProgram);
    if (!execution) {
        LOAD(CreateShader); LOAD(ShaderSource); LOAD(CompileShader);
        LOAD(GetShaderiv); LOAD(GetShaderInfoLog); LOAD(DeleteShader);
        LOAD(CreateProgram); LOAD(AttachShader); LOAD(DetachShader);
        LOAD(LinkProgram); LOAD(GetProgramiv); LOAD(GetProgramInfoLog);
    } else {
        LOAD(UseProgram); LOAD(IsProgram); LOAD(GetProgramiv); LOAD(IsBuffer);
        LOAD(IsTexture); LOAD(GetIntegerv); LOAD(GetIntegeri_v); LOAD(BindBuffer);
        LOAD(GetBufferParameteri64v); LOAD(BindBufferBase); LOAD(BindBufferRange);
        LOAD(BindImageTexture); LOAD(GetUniformLocation); LOAD(DispatchCompute);
        LOAD(MemoryBarrier);
        for (size_t i=0;i<4;++i) {
            auto prefix=std::string("glUniform")+std::to_string(i+1);
            api.iv[i]=resolve<Api::IntUniform>(context,(prefix+"iv").c_str());
            api.uv[i]=resolve<Api::UIntUniform>(context,(prefix+"uiv").c_str());
            api.fv[i]=resolve<Api::FloatUniform>(context,(prefix+"fv").c_str());
        }
        for (size_t i=0;i<3;++i) {
            auto name=std::string("glUniformMatrix")+std::to_string(i+2)+"fv";
            api.mv[i]=resolve<Api::MatrixUniform>(context,name.c_str());
        }
    }
#undef LOAD
    return api;
}
void checked(const Api& api, const char* phase) {
    const auto error=api.GetError();
    // Never allocate on a successful destructive operation: its tracked name
    // must be cleared immediately, before any later failure can retain it.
    if(error)throw std::runtime_error(std::string(phase)+": OpenGL error "+std::to_string(error));
}
std::string driver_log(const Api& api,U object,bool shader) {
    I size=0;
    (shader?api.GetShaderiv:api.GetProgramiv)(object,info_log_length,&size);
    size=std::clamp(size,I(1),I(1024*1024));
    std::vector<char> text(static_cast<size_t>(size)+1,0);
    I written=0;
    (shader?api.GetShaderInfoLog:api.GetProgramInfoLog)(object,size,&written,text.data());
    return std::string(text.data(),static_cast<size_t>(std::clamp(written,I(0),size)));
}
void matching_program(const gp_gl_compute_context& context,uintptr_t native,
                      const gp_gl_compute_program* program) {
    require(program && program->linked && program->name && program->context_id==context.context_id &&
            program->native_context==native,"OpenGL program belongs to another context");
}
void set_uniform(const Api& api,const gp_gl_uniform& uniform,I location) {
    if (location<0) return;
    const auto width=uniform.components-1;
    switch(uniform.type) {
        case GP_GL_UNIFORM_I32: api.iv[width](location,I(uniform.count),static_cast<const I*>(uniform.values)); break;
        case GP_GL_UNIFORM_U32: api.uv[width](location,I(uniform.count),static_cast<const U*>(uniform.values)); break;
        case GP_GL_UNIFORM_F32: api.fv[width](location,I(uniform.count),static_cast<const float*>(uniform.values)); break;
        case GP_GL_UNIFORM_MATRIX_F32: api.mv[width-1](location,I(uniform.count),0,static_cast<const float*>(uniform.values)); break;
        default: throw std::runtime_error("Unsupported OpenGL uniform type");
    }
}
}

extern "C" int gp_gl_compute_link(const gp_gl_compute_context* context,
    const gp_gl_compute_source* sources,size_t count,gp_gl_compute_program** result,
    char* error,size_t capacity) {
    if (result) *result=nullptr;
    report(error,capacity,"");
    Api api; std::unique_ptr<gp_gl_compute_program> owner;
    try {
        const auto native=check_context(context);
        require(result && sources && count && count<=4096,"Invalid OpenGL source package");
        for(size_t i=0;i<count;++i)
            require(sources[i].source && sources[i].source_bytes &&
                sources[i].source_bytes<=size_t(INT32_MAX),"Invalid OpenGL source extent");
        api=load(*context,false); checked(api,"Before compile");
        owner=std::make_unique<gp_gl_compute_program>();
        owner->context_id=context->context_id;owner->native_context=native;
        auto& shaders=owner->shaders;shaders.reserve(count);
        owner->name=api.CreateProgram();
        const U program=owner->name;require(program!=0,"glCreateProgram returned zero");
        for(size_t i=0;i<count;++i) {
            const auto shader=api.CreateShader(compute_shader);
            require(shader!=0,"glCreateShader returned zero"); shaders.push_back(shader);
            const I length=I(sources[i].source_bytes);
            api.ShaderSource(shader,1,&sources[i].source,&length);
            api.CompileShader(shader);
            I ok=0; api.GetShaderiv(shader,compile_status,&ok);
            const std::string label=sources[i].label?sources[i].label:std::to_string(i);
            require(ok!=0,"Compile object "+label+": "+driver_log(api,shader,true));
            checked(api,"Compile object"); api.AttachShader(program,shader);
        }
        api.LinkProgram(program);
        I ok=0; api.GetProgramiv(program,link_status,&ok);
        require(ok!=0,"Combined program link: "+driver_log(api,program,false));
        checked(api,"Program link");
        while(!shaders.empty()) {
            const auto shader=shaders.back();
            api.DetachShader(program,shader);checked(api,"Linked shader detach");
            api.DeleteShader(shader);checked(api,"Linked shader cleanup");
            shaders.pop_back();
        }
        owner->linked=true;
        *result=owner.release();
        return 1;
    } catch(const std::exception& e) { report(error,capacity,e.what()); }
      catch(...) { report(error,capacity,"Unknown native OpenGL link failure"); }
    if(owner) {
        // A failed compile/link can still own driver objects. Reuse checked
        // owning-context release; never discard an unproven cleanup outcome.
        // Returning this opaque owner does NOT make it an executable program.
        auto* pending=owner.release();
        char cleanup_error[512]{};
        if(!gp_gl_compute_release(context,pending,cleanup_error,sizeof(cleanup_error)))
            *result=pending;
    }
    return 0;
}

extern "C" uint32_t gp_gl_compute_program_name(const gp_gl_compute_program* program) {
    return program?program->name:0;
}
extern "C" int gp_gl_compute_release(const gp_gl_compute_context* context,
    gp_gl_compute_program* program,char* error,size_t capacity) {
    report(error,capacity,"");
    if(!program) return 1;
    try {
        const auto native=check_context(context);
        require(program->context_id==context->context_id && program->native_context==native,
            "OpenGL cleanup owner belongs to another context");
        const auto fn=resolve<Api::DeleteProgramT>(*context,"glDeleteProgram");
        const auto get_error=resolve<Api::GetErrorT>(*context,"glGetError");
        require(!get_error(),"OpenGL error pending before program release; owner retained");
        if(program->name) {
            fn(program->name);
            require(!get_error(),"OpenGL program release failed; owner retained");
            program->name=0;program->linked=false;
        }
        if(!program->shaders.empty()) {
            const auto shader_fn=resolve<Api::DeleteShaderT>(*context,"glDeleteShader");
            while(!program->shaders.empty()) {
                shader_fn(program->shaders.back());
                require(!get_error(),"OpenGL shader release failed; owner retained");
                program->shaders.pop_back();
            }
        }
        delete program; return 1;
    } catch(const std::exception& e) { report(error,capacity,e.what()); }
      catch(...) { report(error,capacity,"Unknown native OpenGL release failure"); }
    return 0;
}

extern "C" int gp_gl_compute_submit(const gp_gl_compute_context* context,
    const gp_gl_compute_schedule* schedule,gp_gl_compute_receipt* receipt,
    char* error,size_t capacity) {
    gp_gl_compute_receipt local{0,0,SIZE_MAX,0};
    report(error,capacity,"");
    try {
        const auto native=check_context(context);
        require(receipt && schedule && schedule->stages && schedule->stage_count,
                "OpenGL schedule/receipt missing");
        require((!schedule->storage_count || schedule->storage) &&
                (!schedule->image_count || schedule->images),"OpenGL resource list missing");
        auto api=load(*context,true); checked(api,"Before schedule preflight");
        I storage_limit=0,image_limit=0,alignment=0;
        api.GetIntegerv(max_storage_bindings,&storage_limit);
        api.GetIntegerv(max_image_units,&image_limit);
        api.GetIntegerv(storage_alignment,&alignment);
        require(storage_limit>0 && image_limit>0 && alignment>0,"Invalid OpenGL compute resource limits");
        std::array<I,3> limits{};
        for(U axis=0;axis<3;++axis) api.GetIntegeri_v(max_groups,axis,&limits[axis]);
        std::set<U> storage_slots,image_slots;
        for(size_t i=0;i<schedule->storage_count;++i) {
            const auto& b=schedule->storage[i];
            require(b.binding<U(storage_limit) && storage_slots.insert(b.binding).second &&
                b.buffer && api.IsBuffer(b.buffer),"Invalid/duplicate OpenGL storage binding");
            require(b.offset<=size_t(INTPTR_MAX) && b.bytes<=size_t(INTPTR_MAX) &&
                (b.bytes || !b.offset) && b.offset%size_t(alignment)==0,
                "Invalid OpenGL storage binding range/alignment");
            api.BindBuffer(storage_buffer,b.buffer);
            int64_t allocated=0; api.GetBufferParameteri64v(storage_buffer,buffer_size,&allocated);
            require(allocated>0 && b.offset<=uint64_t(allocated) &&
                b.bytes<=uint64_t(allocated)-b.offset,"OpenGL storage range exceeds allocation");
        }
        for(size_t i=0;i<schedule->image_count;++i) {
            const auto& b=schedule->images[i];
            require(b.unit<U(image_limit) && image_slots.insert(b.unit).second &&
                b.texture && api.IsTexture(b.texture) && b.level>=0 && b.layer>=0 &&
                b.layered<=1 && b.access>=0x88B8 && b.access<=0x88BA && b.format,
                "Invalid/duplicate OpenGL image binding");
        }
        std::vector<std::vector<I>> locations(schedule->stage_count);
        using UniformShape=std::array<U,3>;
        using UniformSet=std::map<std::string,UniformShape>;
        std::map<U,UniformSet> program_uniforms;
        for(size_t i=0;i<schedule->stage_count;++i) {
            const auto& s=schedule->stages[i];
            matching_program(*context,native,s.program);
            require(s.boundary==GP_GL_COMPLETE_INVOCATION,
                "Unverified invocation continuation: complete compiler-owned carried-state ABI required");
            require(api.IsProgram(s.program->name),"OpenGL program no longer exists");
            I linked=0; api.GetProgramiv(s.program->name,link_status,&linked);
            require(linked!=0,"OpenGL schedule contains unlinked program");
            for(size_t axis=0;axis<3;++axis)
                require(limits[axis]>0 && s.groups[axis]>0 && s.groups[axis]<=U(limits[axis]),
                    "OpenGL dispatch group count exceeds device limits");
            require(!s.uniform_count || s.uniforms,"OpenGL uniform list missing");
            std::set<std::string> names;
            UniformSet uniform_set;
            for(size_t u=0;u<s.uniform_count;++u) {
                const auto& value=s.uniforms[u];
                require(value.name && *value.name && value.values && value.count &&
                    value.count<=U(INT32_MAX) && value.components>=1 && value.components<=4 &&
                    value.type>=GP_GL_UNIFORM_I32 && value.type<=GP_GL_UNIFORM_MATRIX_F32 &&
                    (value.type!=GP_GL_UNIFORM_MATRIX_F32 || value.components>=2) &&
                    names.insert(value.name).second,"Invalid/duplicate OpenGL uniform");
                const auto location=api.GetUniformLocation(s.program->name,value.name);
                require(location>=0 || value.optional,std::string("Required OpenGL uniform missing: ")+value.name);
                locations[i].push_back(location);
                uniform_set.emplace(value.name,UniformShape{U(value.type),value.components,value.count});
            }
            const auto prior=program_uniforms.find(s.program->name);
            require(prior==program_uniforms.end() || prior->second==uniform_set,
                "Repeated OpenGL program must declare the same complete scheduled uniform set in every stage");
            program_uniforms.emplace(s.program->name,std::move(uniform_set));
        }
        checked(api,"Schedule preflight");
        // Uniform type/size compatibility and image-format errors must also be
        // discovered before any dispatch, even for a later stage. Apply all
        // uniform sets once for validation, then reapply at their actual stage.
        for(size_t i=0;i<schedule->stage_count;++i) {
            const auto& s=schedule->stages[i]; api.UseProgram(s.program->name);
            for(size_t u=0;u<s.uniform_count;++u) set_uniform(api,s.uniforms[u],locations[i][u]);
            checked(api,"Uniform preflight");
        }
        auto bind_resources=[&] {
            for(size_t i=0;i<schedule->storage_count;++i) {
                const auto& b=schedule->storage[i];
                if(b.bytes) api.BindBufferRange(storage_buffer,b.binding,b.buffer,P(b.offset),P(b.bytes));
                else api.BindBufferBase(storage_buffer,b.binding,b.buffer);
            }
            for(size_t i=0;i<schedule->image_count;++i) {
                const auto& b=schedule->images[i];
                api.BindImageTexture(b.unit,b.texture,b.level,b.layered,b.layer,b.access,b.format);
            }
        };
        bind_resources(); checked(api,"Resource preflight");
        for(size_t i=0;i<schedule->stage_count;++i) {
            local.failed_stage=i;
            require(check_context(context)==native,"OpenGL context changed during schedule");
            const auto& s=schedule->stages[i];
            api.UseProgram(s.program->name); bind_resources();
            for(size_t u=0;u<s.uniform_count;++u) set_uniform(api,s.uniforms[u],locations[i][u]);
            if(s.barrier_before) api.MemoryBarrier(s.barrier_before);
            checked(api,"Before stage dispatch");
            ++local.dispatch_calls;
            api.DispatchCompute(s.groups[0],s.groups[1],s.groups[2]);
            checked(api,"Stage dispatch (outcome may be partial; no replay)");
            if(s.barrier_after) api.MemoryBarrier(s.barrier_after);
            checked(api,"After stage barrier (outcome may be partial; no replay)");
            ++local.stages_submitted;
        }
        local.failed_stage=SIZE_MAX; *receipt=local; return 1;
    } catch(const std::exception& e) { report(error,capacity,e.what()); }
      catch(...) { report(error,capacity,"Unknown native OpenGL submission failure"); }
    local.dispatch_may_have_run=local.dispatch_calls?1:0;
    if(receipt) *receipt=local;
    return 0;
}

extern "C" int gp_gl_frame_read_rgba8(const gp_gl_compute_context* context,
    int32_t x,int32_t y,uint32_t width,uint32_t height,uint64_t max_bytes,
    uint64_t* tensor_handle,char* error,size_t capacity) {
    if(tensor_handle)*tensor_handle=0;
    report(error,capacity,"");
    try {
        check_context(context);
        require(tensor_handle && width && height && max_bytes,
                "Invalid frame readback extent or output");
        require(x>=0 && y>=0 && uint64_t(x)+width<=uint32_t(INT32_MAX) &&
                uint64_t(y)+height<=uint32_t(INT32_MAX),
                "Frame readback rectangle is outside GL coordinate range");
        require(width<=uint32_t(INT32_MAX) && height<=uint32_t(INT32_MAX) &&
                width<=UINT64_MAX/4/height && uint64_t(width)*height*4<=max_bytes,
                "Frame readback exceeds host quota");
        using GetInteger=void(GP_GL_CALL*)(uint32_t,int32_t*);
        using PixelStore=void(GP_GL_CALL*)(uint32_t,int32_t);
        using ReadPixels=void(GP_GL_CALL*)(int32_t,int32_t,int32_t,int32_t,
                                          uint32_t,uint32_t,void*);
        using GetError=uint32_t(GP_GL_CALL*)();
        auto get_integer=resolve<GetInteger>(*context,"glGetIntegerv");
        auto pixel_store=resolve<PixelStore>(*context,"glPixelStorei");
        auto read_pixels=resolve<ReadPixels>(*context,"glReadPixels");
        auto get_error=resolve<GetError>(*context,"glGetError");
        require(get_error()==0,"OpenGL error before frame readback");
        constexpr uint32_t rgba=0x1908,ubyte=0x1401;
        constexpr std::array<uint32_t,4> pack_names={0x0D05,0x0D02,0x0D04,0x0D03};
        std::array<int32_t,4> previous{};
        for(size_t i=0;i<pack_names.size();++i)get_integer(pack_names[i],&previous[i]);
        require(get_error()==0 && (previous[0]==1||previous[0]==2||
                previous[0]==4||previous[0]==8),"Cannot inspect GL pack state");
        std::vector<uint8_t> pixels(static_cast<size_t>(uint64_t(width)*height*4));
        pixel_store(pack_names[0],1);
        for(size_t i=1;i<pack_names.size();++i)pixel_store(pack_names[i],0);
        const auto set_error=get_error();
        if(set_error) {
            for(size_t i=0;i<pack_names.size();++i)pixel_store(pack_names[i],previous[i]);
            throw std::runtime_error("Cannot set GL pack alignment");
        }
        read_pixels(x,y,int32_t(width),int32_t(height),rgba,ubyte,pixels.data());
        const auto read_error=get_error();
        for(size_t i=0;i<pack_names.size();++i)pixel_store(pack_names[i],previous[i]);
        const auto restore_error=get_error();
        require(!read_error && !restore_error,"OpenGL frame readback failed");
        const uint64_t shape[3]={height,width,4};
        const auto tensor=nodus_tensor_create(NODUS_DTYPE_U8,3,shape);
        require(tensor,"Frame tensor allocation failed");
        if(nodus_tensor_write(tensor,0,pixels.data(),pixels.size())!=
           static_cast<int64_t>(pixels.size())) {
            nodus_tensor_destroy(tensor);
            throw std::runtime_error("Frame tensor write failed");
        }
        *tensor_handle=tensor;
        return 1;
    }catch(const std::exception& failure) {
        report(error,capacity,failure.what());
        return 0;
    }
}
