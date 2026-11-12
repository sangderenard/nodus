# Radio as one independent Nodus table

Radio currently has two registered native tool rows in one headless Nodus
table, in this order:

1. `radio.system_output` reads new default-speaker loopback frames and uses
   SpeakToMe's existing native WASAPI/fftfree scope to produce spectral bands.
2. `radio.udp_broadcast` sends that frame over a nonblocking UDP socket.

The table has `ExecMode::Free` and an `Opaque` row-stack contract. Its rows
exchange one typed frame pointer only within the same ordered table iteration;
no Nodus edge or cross-thread handoff intervenes. The WASAPI library retains
its own hardware capture thread. Nodus's persistent Free worker owns analysis,
row ordering, and UDP delivery after **one** initial `nodus_headless_step`.
The canvas needs no further ticks. A graph save/load records the module's Free
execution mode and stack contract in a `MODULE_POLICY` trailer; callers must
register/configure the Radio rows before loading such a graph.

The native executable is `nodus_radio`. It takes the built
`openrgb_audio_scope.dll`, the existing fftfree `fft_cffi.dll`, a destination,
and optional duration/band/FFT settings:

```powershell
nodus_radio.exe <openrgb_audio_scope.dll> <fft_cffi.dll> 127.0.0.1 23999 0 5 1024
```

Duration 0 runs until Ctrl+C. Default bands = 5 and FFT size = 1024; the
overlap is 4, and analyzed frequencies span 40–18000 Hz on the existing
scope's logarithmic scale. The CLI prints the UDP endpoint and once-per-second
capture rate, frame count, sent count, and drop count. The scope DLL must
include `openrgb_audio_scope_frame_count`, which distinguishes a fresh audio
window from its cached FFT snapshot.

UDP packet v1 is little-endian, 28 + 4 × bands bytes:

| Offset | Type | Meaning |
|---:|---|---|
| 0 | u32 | magic `0x4441524e` (`NRAD` bytes on little-endian) |
| 4 | u32 | version = 1 |
| 8 | u32 | band count |
| 12 | u32 | sample rate |
| 16 | u64 | captured stereo-frame index; strictly increasing |
| 24 | f32 | full-window RMS |
| 28 | f32[band count] | logarithmic spectral power, normalized 0–1 by the scope |

The output socket is nonblocking. If a datagram cannot be sent, Radio counts
it as dropped and continues with the next live frame. There is no inbound
socket and no manager handoff between rows. UDP itself has no delivery or
authentication guarantee; use a local destination unless the surrounding
network supplies the needed controls.

On this machine (Windows, 48 kHz loopback, five bands), a five-second local
receiver run with FFT size 1024 received 293 packets. Median interpacket gap
was 15.7 ms, 95th percentile 31.6 ms, maximum 32.7 ms, with no reported
send drops. A separate four-second run verified 149 packets with strictly
increasing frame indices, valid band payloads, and 50 ms shutdown. These are
measurements of the current host, not hard real-time guarantees.

Media-file playback, linear-band mapping, and REST/MCP subscription management
are separate Radio source/control additions. This path supplies the default
system-output source and fftfree analysis through a real Nodus Free table.

## Dedicated WAV source (no system-output capture)

Pass a WAV filename after FFT size and optionally `1` to loop:

```powershell
nodus_radio.exe <scope.dll> <fft_cffi.dll> 127.0.0.1 <relay-port> 0 5 1024 <approved.wav> 1
```

This selects `radio.file_output` in the same Free table. SDL3 decodes the WAV,
converts it to stereo float32, and plays it through an SDL audio stream. Only
that stream's submitted samples feed the existing scope ring/fftfree analysis;
WASAPI capture is never started. NRAD v1 remains spectral telemetry, not an
audio stream. The browser cannot reconstruct or listen to music from NRAD v1.

The file and converted buffer are each bounded to 64 MiB. The producer keeps
approximately two FFT windows queued to the audio device, avoiding an unbounded audio
queue. The reported frame index counts submitted file samples, not hardware
DAC completion; analysis can lead audible output by this small queue plus
hardware buffering. Looping keeps the frame index monotonic. Non-looping stops
at EOF. The source requires a functioning SDL playback device.

`radio_udp_source_port=<port>` identifies the bound sender port for the service's
pinned-peer relay subscription. Reconfigure that subscription after each new
producer launch. No service restart or public subscription is implied by this
CLI. API callers select `nodus_headless_radio_file_source` after configure and
before constructing or launching the table; configuration is not a live switch.

The scope DLL must additionally export `openrgb_audio_scope_feed_stereo`.
Its external-feed and WASAPI modes are mutually exclusive. This reuses the
same analyzer, not a second FFT implementation.

### Verification, 2026-11-12

Native Release build passed with existing vcpkg dependencies (manifest install
OFF to avoid a failing remote CMake download). A three-second real Tempest WAV
playback received 580 valid NRAD packets, strictly increasing sample indices,
nonzero finite RMS, matching reported source port, and zero sender drops.
Shutdown took 79 ms. `tools/test_radio_file_udp.py` retains this bounded test;
it is audible and must be run with an approved file. Scope feed regression
also checks rate mismatch, NaN rejection, and refusal to start loopback on an
external-fed scope. These results do not establish live site publication.

Tempest was generated with unchanged `spectral-analyzer/demo_song_wav.py` at
`da1116dd46a43085d78b45476879789e7a783bb5` using the existing Python runtime:
12 seconds, 48000 Hz, float32 mono WAV, peak 0.92. The original score is covered
by that repository's MIT license. Output and its license stay in untracked user
assets, not this repository. Its generator docstring says PCM16, but its actual
writer emits float32; SDL's WAV conversion supports the real emitted format.

## Opt-in PCM backend output

A final CLI argument selects a **separate private PCM destination**:

```powershell
nodus_radio.exe <scope.dll> <fft.dll> 127.0.0.1 <NRAD-port> 0 5 1024 <wav> 1 <PCM-port>
```

This file-source mode uses a third registered row, `radio.pcm_rtp`, in the same
Free table. The source publishes the actual stereo samples into the canonical
canvas/table-edge named output `radio.pcm`; the RTP row consumes that FIFO. It
is not a spectrum reconstruction, private queue replacement, or new router.
The FIFO is 16 fixed-size blocks, keeps the newest 8 on overflow, and records
refused/overwritten packets. `nodus_headless_publish_output_token` is the narrow
native writer operation on an existing named output, using the ordinary table
edge publication API. Declare/bind before starting persistent execution.

Initial scope: WAV sources, 256-frame hops (1024 FFT / overlap4), stereo48kHz.
SDL performs necessary resampling before the samples enter the scope/PCM FIFO.
The 2064-byte internal block carries first-frame index, rate, valid frame count,
and 256 interleaved float32 stereo frames. The wire uses standard RTPv2 dynamic
payload96, L16/48000/2, signed16 big-endian samples. RTP timestamps increment in
sample frames; initial sequence, timestamp and SSRC are randomized. Typical
payload1024 bytes + RTP header12 bytes fits a normal Ethernet MTU. Clipping and
16-bit quantization are explicit; NRAD v1 remains unchanged. A partial final
file block is a shorter valid RTP payload (the initial backend's fixed-block
receiver may discard it; pad/flush policy is not silently invented).

The sender binds and targets loopback only. It prints its source port and SSRC
for a backend to pin, and separate `pcm_packets` / `pcm_dropped` counters.
RTP itself is not encrypted or authenticated. No public endpoint, firewall rule,
account grant, multicast, or automatic public subscription was added. Existing
backend policy must authorize/allocate a subscription before opening its port.

### Current native UDP route, not the optional SQL send quota

The current `tools/udp_router/udp_router.cpp` in canonical Nodus owns UDP sockets
and greedily drains them into bounded per-destination BATCH frames for local TCP
service clients. OPEN/CLOSE/SEND ownership is per connection. It has no600/minute
rate gate. Python `relay.py` has a configurable600/minute default only for its
own authenticated outgoing send API; this does **not** restrict native incoming
PCM or the native service BATCH interface. No quota change or alternate router
is needed for this path.

`tools/test_radio_pcm_router.py` runs the real WAV source, real canvas FIFO,
RTP row, unchanged native router and BATCH subscription in isolated loopback
processes. It explicitly selects `SDL_AUDIODRIVER=dummy`, so it is silent. A
three-second run received576 valid PCM packets with monotonic timestamps and
sequence, stableSSRC, nonzero stereo data and expected mono-to-stereo mapping.

A separate four-second integration with Turing's actual backend `Subscription`
(at5d8597c7) observed756 sent, zero native drops;738 accepted by the subscriber,
zero invalid/foreign/rejected and zero resyncs. The caller-clocked consumer
returned756 blocks including18 silence concealments and reported10 late packets.
These startup/clock/end-of-stream results are not a lossless or production PA
latency guarantee. The receiver needs caller clock/lifetime management; native
RTP currently has no RTCP BYE/end-of-stream message. No live site deployed.
