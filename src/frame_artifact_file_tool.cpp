#include "frame_artifact_file_tool.h"
#include "frame_tensor_packet_tool.h"
#include "tool_registry.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "value_types.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace nodus::gpu {
namespace {
constexpr std::uint64_t kReceiptWords=4;
std::string filename(std::uint64_t generation) {
    std::ostringstream out;
    out << "frame-" << std::setw(16) << std::setfill('0') << generation << ".nfrm.pack";
    return out.str();
}
}

FrameArtifactFileTool::FrameArtifactFileTool(std::filesystem::path admitted_job_directory,
    std::uint64_t max_compressed_bytes)
    :directory_(std::filesystem::weakly_canonical(admitted_job_directory)),
     max_bytes_(max_compressed_bytes) {
    if(!max_bytes_||!std::filesystem::is_directory(directory_))
        throw std::invalid_argument("admitted frame job directory must exist");
    filesystem_=gp_mem_backend_create_filesystem(directory_.string().c_str(),0);
    if(!filesystem_)throw std::runtime_error("job filesystem backend unavailable");
}
FrameArtifactFileTool::~FrameArtifactFileTool() {
    if(filesystem_)gp_mem_backend_release(filesystem_);
}
std::filesystem::path FrameArtifactFileTool::file_for(std::uint64_t generation) const {
    return directory_/filename(generation);
}

void FrameArtifactFileTool::execute_stack(ToolStackContext& context) {
    if(!context.stack.raw)return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle payload{},control{};
    if(!raw_stack_pop_abstract_tensor(stack,payload))return;
    if(!raw_stack_pop_abstract_tensor(stack,control)) {
        raw_stack_push_abstract_tensor(stack,payload);return;
    }
    auto restore=[&] {
        raw_stack_push_abstract_tensor(stack,control);
        raw_stack_push_abstract_tensor(stack,payload);
        context.succeeded=false;
    };
    NodusTensorDesc pd{},cd{};
    std::array<std::uint64_t,1> request{};
    if(nodus_tensor_describe(control.id,&cd)!=NODUS_OK ||
       cd.dtype!=NODUS_DTYPE_U64 || cd.rank!=1 || cd.shape[0]!=1 ||
       nodus_tensor_read(control.id,0,request.data(),sizeof(request))!=sizeof(request) ||
       nodus_tensor_describe(payload.id,&pd)!=NODUS_OK ||
       pd.dtype!=NODUS_DTYPE_U8 || pd.rank!=1 || !pd.total_bytes ||
       pd.total_bytes>max_bytes_ ||
       !request[0] || file_for(request[0]).empty()) {
        error_="expected generation U64[1] and nonempty compressed U8 span";
        restore();return;
    }
    const auto target=file_for(request[0]);
    if(std::filesystem::exists(target)) {
        error_="frame generation already committed";
        restore();return;
    }
    std::vector<std::uint8_t> bytes;
    try {bytes.resize(static_cast<std::size_t>(pd.total_bytes));}
    catch (...) {error_="compressed frame exceeds available host memory";restore();return;}
    if(nodus_tensor_read(payload.id,0,bytes.data(),bytes.size())!=
       static_cast<std::int64_t>(bytes.size())) {
        error_="compressed frame tensor cannot be read";restore();return;
    }
    const auto* vt=gp_mem_backend_get_vtable(filesystem_);
    if(!vt||!vt->alloc||!vt->copy_to_backend) {
        error_="filesystem backend cannot write";restore();return;
    }
    auto buffer=vt->alloc(filesystem_,bytes.size(),1);
    if(!buffer) {error_="filesystem buffer allocation failed";restore();return;}
    const auto native_path=reinterpret_cast<const char*>(gp_mem_backend_native_handle(buffer));
    if(!native_path || !vt->copy_to_backend(buffer,0,bytes.data(),bytes.size())) {
        error_="filesystem buffer write failed";
        gp_mem_backend_release(buffer);restore();return;
    }
    const std::filesystem::path temporary(native_path);
    gp_mem_backend_release(buffer); // Windows cannot rename the open FILE*.
    // The backend's generated file and final target are in the same admitted
    // directory. Rename is the publication point; the host then records status.
    std::error_code ec;
    std::filesystem::rename(temporary,target,ec);
    if(ec) {
        std::error_code cleanup;
        std::filesystem::remove(temporary,cleanup);
        error_="frame file publication failed: "+ec.message();restore();return;
    }
    const std::uint64_t receipt[kReceiptWords]={0,request[0],
        static_cast<std::uint64_t>(bytes.size()),1};
    const std::uint64_t shape[1]={kReceiptWords};
    const auto output=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!output || nodus_tensor_write(output,0,receipt,sizeof(receipt))!=sizeof(receipt) ||
       !raw_stack_push_abstract_tensor(stack,{output})) {
        if(output)nodus_tensor_destroy(output);
        // The file may already be visible. Never restore and replay this input.
        error_="frame committed but receipt publication failed";
        context.succeeded=false;return;
    }
    error_.clear();
}

bool register_frame_artifact_rows(ToolRegistry& registry,
    const std::string& digest,const std::filesystem::path& directory,
    std::uint32_t color_encoding,std::uint32_t origin,
    std::uint64_t max_packet_bytes,std::uint64_t max_compressed_bytes) {
    if(digest.size()!=64||color_encoding>2||origin>1||
       !max_packet_bytes||!max_compressed_bytes)return false;
    for(char c:digest)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    if(!std::filesystem::is_directory(directory))return false;
    const auto resolved=std::filesystem::weakly_canonical(directory);
    ToolRegistry::Entry packet;
    packet.id="nodus.frame.packet."+digest;
    packet.name="Frame tensor packet";
    packet.factory=[color_encoding,origin,max_packet_bytes] {
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new FrameTensorPacketTool(color_encoding,origin,max_packet_bytes),
            [](ITool* value){delete value;});
    };
    ToolRegistry::Entry file;
    file.id="nodus.frame.file."+digest;
    file.name="Frame artifact file";
    file.factory=[resolved,max_compressed_bytes] {
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new FrameArtifactFileTool(resolved,max_compressed_bytes),
            [](ITool* value){delete value;});
    };
    // Both names are reserved before either registration so a partial pair
    // cannot replace an existing row. Registry mutation is host-only.
    if(registry.find(packet.id)||registry.find(file.id))return false;
    if(!registry.register_tool(std::move(packet)))return false;
    if(!registry.register_tool(std::move(file))) {
        registry.unregister_tool("nodus.frame.packet."+digest);
        return false;
    }
    return true;
}
}
