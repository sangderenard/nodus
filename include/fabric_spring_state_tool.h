#pragma once
// The row sources: module_library/source/tools/fabric_spring_stage.cpp (the four stage rows) and
// module_library/source/tools/fabric_spring_state.cpp (the program row).
// The fabric row (spring.network.batched, turing src/compiler/spring_network_nodus_row.cpp.in) on the tensor state
// library: its state lives at addressed library columns and each tick runs as library rows in a KernelPipelineTool
// schedule. Research case 2a1563063ca34358899d5cb971cdf521 (ticket e5b89702a56b48e2be25d2b8b191eff9).
//
// Library layout from a base address (one card): 0..26 are the law's bindings exactly as the row's 27 buffers
// (0 anchor, 1 contact enabled, 2 contact height, 3 drag, 4 dt, 5..8 edge masks, 9 external acceleration, 10 inverse
// mass, 11..14 neighbour positions, 15 occupied, 16 pinned, 17 position, 18..21 rest offsets, 22 stiffness,
// 23 velocity, 24/25 next position/velocity, 26 U32 lane count), then 27 U32 neighbour table [node slots, 4],
// 28 F64[8] contact {floor on, floor y, table on, table y, min x, max x, min z, max z}, 29 F64[1] substep dt,
// 30/31 the committed position and velocity.
//
// Stage rows (CPU, work U64[4] {card, base, lanes, generation}, receipt U64[9]): the row's own host code, unchanged:
//   nodus.fabric.spring.begin     committed 30/31 -> working 17/23
//   nodus.fabric.spring.prepare   dt column, contact columns from the working position, neighbour gather
//   nodus.fabric.spring.feedback  finite check of 24/25, then 24/25 -> 17/23
//   nodus.fabric.spring.commit    working 17/23 -> committed 30/31
// The law between prepare and feedback is Turing's LLVM row (nodus.fabric.dispatch.cpu) or the GLSL/glslang shader
// as a GpuShaderJobTool (nodus.shader.turing.spring_network_step.vulkan.<device>). A failed stage stops the schedule
// and the committed state is untouched, as the row never replays a tick.
#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nodus::gpu {

constexpr int kFabricNetworks = 8, kFabricNodes = 256, kFabricAxes = 3, kFabricFields = 38;
constexpr int kFabricNodeSlots = kFabricNetworks * kFabricNodes, kFabricBatch = kFabricNodeSlots * kFabricAxes;
enum class FabricStage : std::uint32_t { Begin = 1, Prepare = 2, Feedback = 3, Commit = 4 };

// What the stage rows, the program row and the registrar share (library layout above, receipts and program identities).
inline constexpr std::uint64_t kStageProgram = 0x66616272696331ull;   // "fabric1": the four host stages
inline constexpr std::uint64_t kLawProgram = 0x737072696e6731ull;     // "spring1": the SymPy law, CPU or GPU product
inline constexpr std::uint64_t kNeighbors = 27, kContact = 28, kDt = 29, kCommittedP = 30, kCommittedV = 31, kAddresses = 32;

inline std::array<std::uint64_t, 9> stage_receipt(std::uint64_t status, const std::array<std::uint64_t, 4>& w, std::uint64_t ns) {
    return {status, UINT64_MAX, w[2], ns, w[3], w[1], 0, ns, status == 0 ? 1ull : 0ull};
}

struct FabricSpringProducts {
    std::string llvm_dll;                    // spring_network_step_batched__tick.dll
    std::string llvm_entry = "spring_network_step_batched__tick";
    std::vector<std::uint32_t> vulkan_spirv; // spring_network_step.direct-glsl.spv (glslang build of the GLSL)
    std::uint32_t local_size_x = 64;
};

// Registers the four stage rows, the CPU law row, the GPU law shader jobs, one pipeline per backend
// (nodus.pipeline.fabric.spring.cpu, ...vulkan.<device>) and the program row `row_id`, whose backend is chosen by
// `backend` ("cpu" or "vulkan.<device>"). Returns the ids registered, for the owner to unregister.
bool register_fabric_spring(ToolRegistry& registry, TensorStateLibraryTool& library, const FabricSpringProducts& products,
                            const std::string& row_id, const std::string& backend, std::uint64_t card, std::uint64_t base,
                            std::vector<std::string>* registered);

}  // namespace nodus::gpu
