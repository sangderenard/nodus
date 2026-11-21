# Ray kernel jobs and TCP off ramps (2026-11-13 UTC)

## Current executable boundary

The kernel library already accepts shader contracts and ordered U64 work
tokens. `KernelLibrarySession` now owns the existing tensor-state-library row
while registered GPU and pipeline factories refer to it; its destructor drops
the factories before destroying their library. The joined 512×512 wave trial
passed on all three Vulkan devices through that owner. This is process
lifetime management, not a new scheduler or optical algorithm.

The native `TcpByteStreamTool` is a reusable two-port row. It accepts U8
AbstractTensor bytes on its Argument port and emits received U8 tensors on its
Return port. It uses the existing loopback-only `ContractListener` Hello
admission (72-byte version-2 Hello, role 1 or 2 and nonzero participant ID),
then u64 little-endian payload length followed by that many bytes. The row
uses nonblocking bounded sends/receives, a configurable frame limit and queue
depth; a full outbound queue restores its input, and a full inbound queue
stops reads to apply TCP backpressure. A partial queued send lost with a
connection marks delivery uncertain rather than replaying it. The row does
not interpret ray data or bind a public network port.

`nodus_headless_register_tcp_byte_stream` admits it into the existing headless
tool registry. The graph can bind the ID as a row and connect its ports to
ordinary typed edges. The direct socket round trip and the headless bind,
step, destroy and re-register tests passed. They do not yet prove a ray job
streamed through a graph edge.

## Optical job contract to connect next

The job ingress must carry the authored launch vectors (`RayIntent`) and a
contract selecting transport configuration, source/program identities,
resource budgets and requested products. Nodus must interpret that contract
into stage rows and work tokens; Spectral owns the stage physics and record
formats. The existing Spectral `RayRecord` already distinguishes STRIKE,
TERMINAL, MISS and FIELD; BDPT has separate vertex, spectral and PDF side
records, and endpoint/integrator paths have distinct pre- and post-film data.
These are distinct off ramps, not different names for the final image.

The requested products should be graph taps after the stage that creates
them: crossing vectors after T1/T2, path records after T3/T4, endpoint and
linear spectral values before integration, and film/image values after it.
Each tap needs job, generation, stage, record layout, count and completion
provenance in its application envelope before the generic TCP row sends its
bytes. An output edge must publish when that stage finishes. The present
`KernelPipelineTool` executes its whole string before returning one summary
receipt, so it cannot itself deliver intermediate products immediately; the
production workflow must schedule the existing component rows separately in
the Nodus graph and attach the taps to their completed edges. Do not infer
availability of an off ramp from the existence of a destination socket.

The current ray contracts are still registered by test fixtures. No live
launch-vector/contract interpreter, optical tap rows, endpoint envelope,
fully coupled T1–T5 schedule, or remote TCP exposure service is claimed by
this change. The existing contract socket is the control plane; the new byte
stream row is a data-plane building block.
