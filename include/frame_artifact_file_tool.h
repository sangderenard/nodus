#pragma once
// One admitted job's compressed-frame sink. The host supplies its resolved
// owner-scoped directory; neither a FIFO token nor shader source may name it.
#include "tool_api.h"
#include "mem_backend.h"
#include <cstdint>
#include <filesystem>
#include <string>

class ToolRegistry;
namespace nodus::gpu {

class FrameArtifactFileTool final : public ITool {
public:
    explicit FrameArtifactFileTool(std::filesystem::path admitted_job_directory,
                                   std::uint64_t max_compressed_bytes);
    ~FrameArtifactFileTool() override;
    FrameArtifactFileTool(const FrameArtifactFileTool&)=delete;
    FrameArtifactFileTool& operator=(const FrameArtifactFileTool&)=delete;

    const char* id_cstr() const noexcept override {return "nodus.frame.artifact.file";}
    const char* name_cstr() const noexcept override {return "Frame artifact file";}
    ToolCaps caps() const override {return ToolCaps::None;}
    int32_t port_count() const override {return 3;}
    ToolPortSpec port_spec(int32_t index) const override {
        if(index==0||index==1)return {ToolPortKind::Argument,1};
        if(index==2)return {ToolPortKind::Return,1};
        return {};
    }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override;

    // The host must call this only after matching the receipt to its admitted
    // SQL job and recording the artifact in that job's status. No client path.
    std::filesystem::path file_for(std::uint64_t generation) const;
    const std::string& last_error() const noexcept {return error_;}
private:
    std::filesystem::path directory_;
    std::uint64_t max_bytes_=0;
    gp_mem_backend_handle_t filesystem_=nullptr;
    std::string error_;
};

// Called by the trusted job host after SQL admission resolves the owner's
// directory. Creates the two rows the canvas can bind by registry id.
bool register_frame_artifact_rows(::ToolRegistry& registry,
    const std::string& admitted_digest,
    const std::filesystem::path& admitted_job_directory,
    std::uint32_t color_encoding,std::uint32_t origin,
    std::uint64_t max_packet_bytes,std::uint64_t max_compressed_bytes);
}
