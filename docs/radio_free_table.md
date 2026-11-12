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
