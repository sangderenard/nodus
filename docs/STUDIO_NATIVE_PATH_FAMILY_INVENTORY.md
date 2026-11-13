# Native transport roots, copied responses and remaining stage boundaries

2026-11-11 UTC, read-only source inventory. This document does not admit
execution or implement a second optical renderer. V1/v2/v3 frozen forward
packages remain untouched. Source hashes for this reading are recorded in
`STUDIO_NATIVE_PATH_FAMILY_SOURCES.tsv`; composition options are part of source
identity and must be frozen before any later native source package is authored.

## Distinct populations: never substitute one for another

| Family | Actual root and native output | Existing copied evidence | Current native staged coverage |
|---|---|---|---|
| Prebuilt forward | Actual baked SourceRec index `ray_id % max(uSourceCount,1)`; native RNG uses original chunk ray ID/seed; packet path deposits scalar images and optionally directional segment aggregates | Forward response rows, exact before/after packet and material/termination snapshots; root replay weight belongs to authored forward population | V1/v2/v3 source candidates only; no verified continuation execution |
| Reactive secondary | Actual stored PendingRay queue slot, packet and remaining bounce budget; native RNG keyed to dispatched queue gid/seed; writes scalar field and response rows when instrumented | Path family2 with native queue index and explicit unavailable original root; no guessed forward parent | No authored runnable native stage package or prebuilt queue drain |
| Legacy cone sensor | `sensor_main` inside ray-field program: cone rays from uSrcPos, native cone RNG/PDF/path weight, immutable forward-field snapshot reads, scalar-volume writes | Not actual completed camera-film samples; no right to relabel those scalar contributions as family4 | No staged package; separate optional legacy path |
| Actual camera | `_GPU_SENSOR_CS`: physical invocation, selected sensor pixel/full sensor origin, sample index and lens/aperture root; throughput/lum arrays and native film/pixel integration | Interval identity, evaluated RHS response copies, optional completed sample/film columns | Existing optional copy hooks, not a native staged producer package |
| Camera NEE | Inline `sample_direct_light` at an actual camera material event; original RNG source choice and original shadow nearest-hit | Actual NEE outcome/source/shadow, direct scalar/bands and PBR RHS copies; optional separate diagnostic ledger | Not an independent root population or existing standalone dispatch |

Scene/material family availability remains independent: opaque, transparent,
and camera volume cannot be inferred from a forward/material link succeeding.
Interior directional segment observation is not interior camera volume
scattering. An unavailable family cannot be filled with zero or derived from
another family's packet power.

## Canonical source and composition boundaries

`spectral-analyzer/demo_pluck_gl.py` defines `_GPU_RAY_FIELD_CS` around3286,
`sensor_main`4170, generic main4289, `reactive_main`4532 and `_GPU_SENSOR_CS`
4725. The ray-field entry distinguishes mode0 forward,1 legacy cone sensor,
2 reactive. Actual camera has its own 16x16 entry around5168, not the ray
field's 128x1 entry.

SensorAccumulator's exact source composition occurs before GL imports around
10256–10297: region/sphere base, optional native opaque PBR, optional volume
extinction, optional separate-band continuation, interval observer, optional
scalar-availability queries, response copies, optional completed samples. The
final `_sensor_shader_source` hash, not only this Python module's hash, is the
identity needed for any native author. A full camera continuation ABI must
inventory all globals added by those selected composers.

The current preview owner has an explicit execution hold before context
creation. At this reading its hold is line488 and context call line581;
anchors, rather than historical line numbers, identify it. Ordinary no-opt-in
behavior must remain unchanged.

## Reactive queue: an exact outstanding producer gap

`append_pending_ray` at4152–4160 increments `pending_count` and stores only when
the returned index is below uPendingCap. The four vec4 payloads are:
origin/flags, direction/remaining budget, packet, and hit normal/triangle ID.
There is no original root ID, original root replay weight, parent bounce or
parent queue lineage. The present `pray_flags` local is loaded but not used
again by `reactive_main`; this does not authorize repurposing its bits.

Reactive initialization uses the actual queue slot, normalizes its existing
direction, copies its packet and remaining budget, and runs original origin
classification. Its loop is `bounce < min(bounces_rem,uMaxBounces)`, not the
forward loop's budget and not camera's inclusive budget. Its scalar deposition
uses `splat_segment_spectral`, not forward multiscale/wave/path_len work. Native
PBR and region composers must be retained when selecting the final source.

The ordinary generic host at7395–7473 runs forward, optional cone sensor, then
reactive. It snapshots original field images before cone sensor reads, uses
the original batches and then reads counter4 after STORAGE|BUFFER_UPDATE
barrier. Reactive work is bounded to min(attempted,capacity), uses seed31415926,
offset0 and the original 128-thread grid, then IMAGE|STORAGE visibility. This
existing generic path does not prove prebuilt completeness.

Prebuilt opt-in at7890–7901 instruments responses and explicitly specializes
the entry to forward only. `specialize_prebuilt_forward` documents that pending
reactive entries remain unprocessed. The same owner cannot claim queue complete
just because some forward response rows were copied.

Exact future requirements:

- Freeze a canonical, explicitly reactive-admitted source, rather than call
  the unreachable helper from a forward-only frozen entry.
- Preserve original queue producer order within each invocation, finish all
  original forward chunks before drain, and record attempted/capacity/stored/
  dispatched/dropped/pending with sticky incompleteness and regression flags.
- Start from actual queue gid and RNG. Carry bounces_rem, bounce, ro/rd/packet,
  currentMedium, native active/primitive globals and exact response state.
  Do not graft forward path_len/multiscale work onto the reactive program.
- Treat missing parent root identity as unavailable. If root lineage is
  required, author an explicit opt-in queue-side provenance copy at the
  original enqueue site, with checked allocation and every edge/reactive
  producer accounted for. Inferring parent from queue index is invalid.
- In the frozen composed source, `studio_record_root_weight` initializes to0
  and is assigned only by forward main. The recorder's reactive helper call
  therefore cannot silently become normalized reactive directional density.
  Root-weight provenance is unresolved, not permission to invent a weight.

All finite queue/row saturation order is schedule-sensitive; a staged schedule
can refine native-allowed atomic ordering without matching an observed run's
queue indices. This does not permit changing queue capacity, early drain or
dropping completeness evidence.

## Legacy cone sensor is a separate field-writing path

The cone sensor samples existing forward textures and direct emitter
connections, then deposits into scalar images. Around7397–7411 the host creates
an immutable texture snapshot before the sensor writes the original field.
A split must preserve that snapshot and its visibility/lifetime, not bind the
live output field as input. Its own ray ID, cone PDF, path weight, RNG, medium
and segment state form a distinct ABI. Forward root weights and camera sample
denominators do not apply. It is not proof of actual camera radiance.

## Actual camera: exact state and lawful candidate seams

Original entry derives physical `linear_id`, selects pixel (possibly random),
keeps full-sensor/tile identity and runs sample loop s<uSamplesPerPixel. Its
RNG formula includes full sensor coordinates, sample, seed and the original
untiled invocation term. Original lens/manifold/aperture root construction
must execute once, with its existing RNG calls. Camera bounces use
`bounce <= uMaxBounces`, different from forward/reactive.

Possible source-preserving cuts, not yet authored/admitted:

1. Original invocation and sample/root initialization.
2. Primary nearest-hit plus immediate primitive/interval identity latches.
3. Original per-band Beer and air-only camera volume integration, with native
   box bounds, jitter, vol_lum/alpha and throughput/lum updates.
4. Original emitter/boundary/material and inline NEE evaluation/continuation.
5. Original completed sample copy, native film filtering, layer accumulation
   and final pixel publication at their existing sample/loop sites.

Required live state includes physical invocation ID, px/sensor_px/full-size
and tile mode, sample and bounce, RNG, nl/active-band path mode, ro/rd,
currentMedium, throughput[8], lum[8], layer_acc[8], emission_via_bsdf, original
best/hit and computed hp, all active-region and primary-hit globals, plus
observer row/index, NEE/air availability, first-recipient position/direction/
medium/material/active bits/prefix and completed component/known arrays.
Any cut inside volume or NEE helper adds its actual local loop/helper state;
it cannot be guessed from output names. No complete camera ABI is claimed yet.

Random-pixel mode can send distinct physical invocations to the same pixel.
`sensor_store` at5145–5152 performs imageLoad+imageStore, not an atomic sum.
Thus slots must be owned by physical invocation, never pixel alone. A later
split must preserve or explicitly resolve the original concurrency contract;
it cannot quietly replace that write with a different averaging algorithm.

Camera volume is marched only for currentMedium<0. Additional authored
achromatic extinction is not a scattering/emission model. Directional
observation inside glass does not add camera volume integration there.
Authored zero alpha/scatter controls establish only their exact disabled
dependency, not general volume completeness.

## Inline NEE and copied camera responses

`sample_direct_light` around5002 consumes the original source-choice RNG,
uses actual area/point source geometry and radiance, runs the original shadow
nearest_hit and returns direct frequency/bands. The optional native PBR
composer adds returned light_direction and evaluates its existing PBR RHS.
There is no extra ray generation or duplicated visibility calculation in
`copy_camera_responses`.

Shadow intersection mutates private sphere/partition/global hit state. Primary
hit identity, loaded material/Tri and response latches must be preserved before
that nested call. A future split should first keep NEE inline with material;
if its graph requires a further cut, preserve the actual candidate index,
source/direction/distance/cosines, occlusion outcome, primary latches, outputs
and RNG at a reviewed helper boundary. Do not redraw, reshadow or rederive it
from a forward directional record.

`studio_camera_responses.py` has12 exact evaluated channels: air-volume add,
area emission, legacy emission, surface-field add, direct-light add, PBR NEE,
Beer factor, boundary factor, legacy transmission, surface survival, PBR
factor and air factor. Each original RHS is passed once to a copy helper and
returned unchanged to the original accumulator/multiplier. Presence includes
known zero, separate from missing/unavailable. Availability uses actual NEE
outcome and scalar-query flags, not reconstructed intensity.

CameraIntervals binding22 and CameraResponses binding24 use matching128-word
rows. Optional separate NEE connection ledger also claims binding22 and
expects the pre-PBR direct-light signature plus its arrival-ledger hooks.
It is not composable by assumption with current interval/PBR packages; an
explicit source/ownership merge would be needed. It remains diagnostic, not
a new source of forward energy or a substitute camera sample.

## Completed samples, RGB and storage gates

`studio_completed_samples.py` writes the actual completed native lum[8] before
film integration and additionally copies the actual original filtered/clamped
film values at the existing layer_acc addition. Its volume/surface component
sums mirror the same evaluated RHS additions. It does not reconstruct transport
or filter measured power/directions. It records termination, known bits and
first recipient. The completed-only layout uses binding25 and removes inactive
binding22/24 blocks, leaving guarded raw-copy paths disabled. All allocations,
capacity/counter/overflow accounting and readback lifetimes remain explicit.

Completed-sample copy currently rejects `separate_field_continuation`: the
default shared-band0 optical path and the explicit separate-band optical
variant are distinct source/estimator identities. The latter is an authored
transport change, not a passive observer switch. Neither variant can silently
borrow the other's denominator, root identity or acceptance receipt.

`studio_scene_application.py` selects only family4 completed camera samples.
It requires exact current scene/transport program/source/camera/lens/film/query
identity, copied first-recipient prefix compatibility, known per-component
values, and suppression after support eviction. Queue completeness requires
observed attempted==stored==dispatched, no drop/pending/sticky failure. Its
only disabled-volume exception applies to the exact authored volume component.
Opaque/glass/volume availability is separate. Native film columns then enter
the existing linear profile, conditional RGB and counts. Response rows, scalar
cells and forward/reactive packet counts are not camera sample denominators.

Current application-ready remains false. The root request still forbids
rendering until the full information path is established. Independent forward
link success cannot close any of these omitted family/source/ownership gates.

## Next concrete authoring work

First complete the protected native forward source/resource/schedule contract,
without turning link success into execution admission. In parallel, freeze
the actual selected camera composer output and a reactive-admitted source,
including source/program/scene/root/query identities. Author their typed native
continuation and copied-state inventories at the seams above. Resolve queue
root provenance and camera raw-storage/ledger binding overlap before claiming
full family support. Use canonical Nodus scheduling/memory owners and exact
original dispatch populations. No Python coordinator or alternative renderer.
