#pragma once

#include <cstdint>

namespace nodus::tensors::kpath {

template <typename Tag, typename Rep = uint32_t>
struct Id final {
  Rep v{};
  constexpr Id() noexcept = default;
  constexpr explicit Id(Rep x) noexcept : v(x) {}
  constexpr explicit operator bool() const noexcept { return v != 0; }
  friend constexpr bool operator==(Id a, Id b) noexcept { return a.v == b.v; }
  friend constexpr bool operator!=(Id a, Id b) noexcept { return a.v != b.v; }
};

struct TokenTag{};
struct NodeTag{};
struct EdgeTag{};
struct ProgramTag{};
struct SchemaTag{};
struct FrameTag{};
struct FontFaceTag{};
struct DeviceTag{};
struct KernelTag{};

using TokenId   = Id<TokenTag>;
using NodeId    = Id<NodeTag>;
using EdgeId    = Id<EdgeTag>;
using ProgramId = Id<ProgramTag>;
using SchemaId  = Id<SchemaTag>;
using FrameId   = Id<FrameTag>;
using FontFaceId= Id<FontFaceTag>;
using DeviceId  = Id<DeviceTag>;
using KernelId  = Id<KernelTag>;

enum class RotDir : int8_t { Neg = -1, Zero = 0, Pos = +1 };
enum class InterpMode : uint8_t { Step = 0, Linear = 1, Cubic = 2, BSpline = 3, PoseLinear = 4, PoseSpline = 5 };
enum class ToolMode : uint8_t { Travel = 0, Draw = 1, Cut = 2, Laser = 3, Extrude = 4, Probe = 5 };

} // namespace nodus::tensors::kpath
