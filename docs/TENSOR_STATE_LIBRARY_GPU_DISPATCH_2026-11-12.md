# Tensor state library and four-worker dispatch trial (2026-11-12)

This is an implementation and measurement receipt, not a claim that a live
canvas table or production service is complete.

## Native path

- `nodus.tensor.state.library` is a registered `ITool` row. It accepts FIFO
  control and tensor payload tokens and returns a receipt token. Its current
  operations are Card, Put, Remove, Status, and ChooseResident. Cards reserve RAM and per-GPU
  quotas; the GPU quota currently limits **simultaneously leased** bytes. It
  does not yet enforce a resident-cache share per card or authenticate a
  caller across processes.
- `nodus.fabric.dispatch.vulkan.0`, `.1`, `.2` are individual native tool
  rows for the three physical Vulkan devices on this host. An LLVM CPU row
  consumes the same addressed tensor library. Each worker returns a timed
  receipt. `EdgeTensorFifo` carries commands and receipts; device buffers
  stay behind the library's address and lease API.
- The state bank uses device-local Vulkan buffers, exact allocation bytes,
  a bounded VRAM cache, LRU eviction and reuse, and host backing. A host
  staging buffer carries host/device transfers. No Python scheduler or
  duplicate spring law is in the dispatch path: the shader is Turing's
  emitted direct GLSL/SPIR-V product and the CPU entry is Turing's LLVM DLL.

## Reproduced observations

Hardware Vulkan inventory: GTX 1660 SUPER, AMD Radeon Graphics, RTX 3060.
The focused tests `vulkan_state_bank_test`, `tensor_state_library_test`, and
`tensor_state_library_tool_test` pass on this host. The bank test confirms a
two-buffer warm cache has zero evictions and the library test confirms GPU
write -> other GPU read and CPU write -> GPU read coherence.

`tools/profile_tensor_dispatch_network.cpp` ran 12 work tokens per worker
concurrently. Each token executes one Turing spring step on a disjoint state
range. The four result FIFOs were collated by one writer into a common sink;
48/48 results arrived. The active range was 4 full 256x1024 fabrics on the
GTX, 604,947 scalar lanes on AMD, 16 full fabrics on RTX, and one full
fabric on LLVM CPU. With the corrected cache, wall time was 1.494 s including
initial transfer of 3.56 GB of tensor state and worker setup.

| Worker | Median lease | Median dispatch | Median total |
|---|---:|---:|---:|
| GTX 1660 SUPER, 4 fabrics | 0.0165 ms | 3.7053 ms | 3.7491 ms |
| AMD Radeon, budget tile | 0.0097 ms | 5.2376 ms | 5.2559 ms |
| RTX 3060, 16 fabrics | 0.0189 ms | 12.0476 ms | 12.0763 ms |
| LLVM CPU, 1 fabric | 0.0193 ms | 8.4956 ms | 11.4887 ms |

The earlier 4-round run showed 87-655 ms median GPU lease time. Two cache
mistakes caused repeated RAM -> VRAM copies: searching for a reusable buffer
evicted otherwise warm entries, and a read-only GPU lease invalidated its own
cached copy whenever host RAM was canonical. Both were corrected. A third
case retained a previous GPU writer as the latest source even after its
contents were flushed to host; clearing that marker after a successful flush
preserves valid read-only copies on other devices.

## Limits and next integration work

- The benchmark repeats the one-step numerical law with pre-gathered neighbor
  feeds. It measures dispatch and cache behavior, **not** a valid evolving
  multi-step cloth simulation. The next compute product must gather neighbors
  from current positions and synchronize before every step.
- The library holds the canonical backing in RAM. Nodus already has a
  filesystem memory backend, but a bounded RAM/disk tier and CPU-page lease
  semantics have not been connected. This run does not represent 10,000
  simultaneous fabrics or a 126 GB position/velocity corpus.
- The four workers are registered Nodus tools, exercised by a native test
  harness with dedicated OS threads. They are not yet installed as a live
  canvas table or service. The library card is local to this process.
- This path uses `dispatch_spirv_backends`, which currently creates a Vulkan
  pipeline and waits for completion per call. Pipeline reuse, batching policy,
  sustained multi-card fairness, and disk spill profiling remain to be done.

## Resident work quantum and alternating-state trial (later 2026-11-12)

The Turing SSA GLSL emitter now accepts declared feed expressions. Its spring
resident product reads each neighbor from the **current position buffer**
through the authored edge mask and row stride. The SymPy/SSA numerical law is
unchanged. The native Vulkan runner records the shader dispatch, position and
velocity feedback copies, and required barriers repeatedly in one command
buffer and waits once at the end of the quantum. A four-step 4x4 grid matched
the independently compiled Turing LLVM law exactly (48 scalar lanes; maximum
observed difference 0). A native GPU row carries an immutable shader work
contract with a maximum number of steps per loadout and an optional target
host-observed dispatch completion time. This is not a physical GPU utilization
measurement. It selects the next quantum from a rolling completion average,
starting with one step when a target is configured. The market-turn deadline
in `ContractNegotiator` is unchanged; it is not a compute quantum.

The network profiler alternated two addressed state ranges per GPU under
explicit VRAM budgets (GTX 800 MiB, AMD 120 MiB, RTX 1536 MiB). The GTX jobs
were four full 256x1024 fabrics each, AMD jobs one 256x512 fabric each, RTX
jobs eight full 256x1024 fabrics each. Its 12-round run completed 48/48
receipts and verified that both ranges on all three GPUs evolved. The banks
reported 289, 283, and 291 evictions respectively, proving that the alternating
states moved through VRAM rather than remaining resident together. With the
library's original global lock around transfers, the same trial took 28.925 s.
Reserving an address under the library lock and moving each GPU's transfer
under its own bank lock reduced the measured wall time to 19.348 s. CPU
median lease time in that run was 0.0187 ms. GPU lease medians remained high
because every alternating range requires synchronous host/device transfers:
roughly 0.20 s on AMD and 1.59 s on RTX. These are measurements of the
current synchronous cache path, not GPU compute time.

One earlier 12-round run stopped on a rejected RTX lease; a subsequent
instrumented 4-round run and 12-round run completed, including the state
checks. The transient refusal has not been assigned a cause. This path still
uses one Vulkan queue per device and synchronous copy/fence waits. A separate
transfer queue, per-buffer completion tokens, prefetch slots, and live canvas
integration remain to be implemented and measured before claiming transfer
overlap or a production scheduler.

The rebuilt Turing resident SPIR-V product and its checked work-contract
manifest drove a four-round native trial after these changes. All 16 receipts
arrived in 4.526 s; the three GPU banks reported 73, 67, and 75 evictions,
and both state ranges on each GPU evolved. Median GPU lease times were
1,062 ms GTX, 190 ms AMD, and 1,482 ms RTX because this deliberately alternated
states across budgets. Median dispatch times were 15.10, 12.38, and 12.83 ms
respectively. These four rounds measure the transfer-pressure case and do not
measure the new warm-preferred offer under sustained market load.

## Resident relending and composite-row boundary

`ChooseResident` ranks ready addressed ranges under a card's device quota:
all fields resident wins first, then the most resident bytes, with FIFO order
for ties. It returns only a choice; the existing work loan remains the authority.
`resident_fabric_offer` applies that choice in `ContractNegotiator::Offer`,
mapping the peer to a device and the opaque payload to a library range. A
fabric offer also carries an absolute steady-clock expiration and a recent
end-to-end completion estimate. If the estimate reaches the expiration, that
job takes priority over cache warmth; among at-risk jobs the largest projected
lateness wins, with FIFO ties. The owner of each payload must provide and
update its completion history. A focused native test submits cold-before-warm
contracts through the existing negotiator and confirms the warm range wins
while both have slack, then the cold contract wins as its expiration nears;
borrowing the warm state does not increment restore transfers. A resident
copy is scored cold if another GPU holds a newer write. The alternating-state
profile above deliberately bypasses this preference to measure the transfer
path under pressure.

The Vulkan runner now keeps a bounded per-device cache of compute pipelines
keyed by exact SPIR-V bytes, entry point, and binding count. The resident
spring test used one created pipeline and one cache hit across two work
quanta. This avoids recreating the pipeline every quantum; it does not prove
driver instruction-cache residency or parallel Vulkan queue performance.

`module_library_actualizer.cpp` already emits a single `ITool` source from a
module's ordered tool rows, with exposed argument/internal/return ports and
inlined plugin sources when available. That is the existing table-to-row
path to use for composite devices. Its generated execution is sequential and
its `tick` is empty; it does not yet lower a canvas's cross-table schedule,
nest tables/canvases, or let a child canvas clock a parent table. The fabric
library and dispatcher are ordinary `ITool` rows and can be composed through
that route once source registration and lifetime/port contracts are supplied.

## Headless canvas and equal-work fanout receipt (2026-11-12 UTC)

The network profiler now places the native state-library-backed GPU and CPU
tools in headless canvas modules with typed abstract-tensor FIFO ports. Nodus's
persistent `ThreadManager` workers execute those modules. A new per-canvas
parallel-worker limit is distinct from the number of rows or queued jobs;
zero keeps the existing automatic sizing. The trial creates independent CPU
fabric jobs on one through six rows and caps active pool lanes to the three
GPU jobs plus that many CPU jobs. No service deployment is implied by this
user-space build.

The apples-to-apples trial uses a 256x256x3 scalar-lane fabric on **every**
device, one Turing-authored spring law, four steps per completed job, and a
fresh Turing LLVM product compiled for exactly 196,608 lanes. The CPU tool
now gathers neighbors from its current position before each step and feeds
the compiled law; it copies the position and velocity outputs back to its
addressed state between steps. A one-round four-step trial returned the same
displayed position sample (2.5499) from all three Vulkan devices and LLVM
CPU. This sample is a spot check, not a full-array parity proof.

With two Vulkan compute queues requested per device and two jobs queued on
each GPU, 12 canvas rounds gave these steady throughputs after excluding the
first round. Jobs/s counts **whole four-step jobs**, not scalar steps:

| CPU worker jobs | Total jobs/s | Median round ms | GPU0 median dispatch ms | GPU1 median dispatch ms | GPU2 median dispatch ms | CPU median dispatch ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 506.1 | 13.70 | 2.30 | 10.48 | 1.52 | 9.51 |
| 2 | 412.9 | 19.43 | 2.31 | 12.87 | 0.91 | 16.5 |
| 4 | 291.6 | 34.88 | 2.48 | 16.24 | 1.60 | about 29.5 |
| 6 | 264.9 | 46.76 | 2.63 | 16.39 | 1.76 | about 41 |

Two repeated 24-round runs in reversed order measured 400.3 and 407.2
jobs/s at two CPU workers, versus 464.9 and 484.4 at one. CPU fanout
therefore degrades this mixed load on this machine. The four-step CPU-only
24-round sweep measured 157.0, 164.6, 165.4, 161.8, 163.1, and 164.2
jobs/s for one through six workers respectively. Two or three CPU workers
give nearly the highest observed CPU-only throughput with much shorter round
latency than six. With one job on each GPU, the 12-round mixed-light trial
measured 441.6, 347.6, 221.2, and 231.7 jobs/s at one, two, four, six CPU
workers. These are workload-specific observations; they do not justify a
fixed global CPU count.

The three GPU modules do execute concurrently in this canvas. In the
one-CPU mixed trial, the six GPU median dispatch durations alone sum to more
than 28 ms, while the median canvas round took 13.7 ms. The round cannot be
a serial execution of all six GPU dispatches. This establishes host-side
overlap and simultaneous device submission; it does not measure hardware
occupancy or prove that queues within one GPU run on independent compute
units. For equal individual 256x256x3 jobs, separate two-job GPU trials at
one/two/four requested queues measured 595/772/719 jobs/s on GTX 1660
SUPER, 214/216/218 on AMD Radeon Graphics (four capped to two available),
and 1059/1158/1137 on RTX 3060. Those one-pass queue results need repeats
before making a queue-count default.

The state library now records an eight-sample rolling completion history by
execution API, device, program identity, and invocation count. The optional
known-peer path in `resident_fabric_offer` uses those exact histories when
every idle compatible GPU has at least three samples. A native market test
verifies that a cold job goes to the fastest measured GPU, its next job goes
to the next measured idle GPU, and a fully resident copy changes the offer.
The test uses synthetic timings: it verifies the decision machinery, not a
production throughput improvement. Queue depth, transfer cost, CPU peers,
and expiry still need a unified measured cost model. No AMD preference is
encoded.

Spectral Analyzer's original T1 BVH intersection shader was adapted directly
from its GLSL source to Vulkan GLSL by replacing eight standalone integer
uniforms with one 32-byte read-only storage block. `glslangValidator -V`
emitted SPIR-V, `spirv-val` accepted it, and reflection verified the nine
SSBO bindings and 64-invocation workgroup. This did **not** use Nodus's
KernelIR-to-SPIR-V translation. The receipt with source and artifact hashes
is in the user-space `spectral-ray-bvh` build directory. The generic native
shader-job row uses the existing Vulkan dispatch and tensor-state leases.

In a 24-round headless canvas trial, each of the three GPUs ran two four-step
256x256x3 fabric jobs and one 16,384-ray T1 job, while one CPU fabric worker
also ran. All 240 row receipts succeeded; each T1 output counted 16,384
hits. Warm median round latency was 15.03 ms. Median T1 dispatch durations
were 0.76 ms (GTX 1660 SUPER), 2.45 ms (AMD Radeon Graphics), and 1.46 ms
(RTX 3060). The ray row's end-to-end times include counter reset and other
state work, and were 12.04, 3.32, and 2.85 ms respectively; on the GTX,
pre-dispatch work (which includes the counter reset) dominated the measured
shader call. The current receipt does not separate reset from any wait or
transfer, so it cannot attribute that overhead further. These timings are from a mixed
load, not an OpenGL-versus-Vulkan comparison. An equivalent native OpenGL T1
trial with the same state, device, and synchronization has not yet been run;
there is no speed claim for the translation. Fountain's particle product is
Turing WGSL plus native CPU and still needs verified artifact adoption.
