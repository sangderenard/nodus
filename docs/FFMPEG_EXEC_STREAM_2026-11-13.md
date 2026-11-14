# FFmpeg as a continuous Nodus exec row (2026-11-13)

`exec_row` now accepts `mode stream` in the host's `NODUS_EXEC_ROWS` declarations. The row starts one child, leaves stdin open across ticks, and accepts successive AbstractTensor byte payloads. It has one argument port and no return port. The existing `mode span` (the default) still runs one child per span and returns its stdout tensor.

Example for 48 kHz stereo signed 16-bit **little-endian** PCM input and AAC HLS output:

```text
row exec.radio.hls
name Radio AAC HLS
program C:/Windows/System32/ffmpeg.exe
directory C:/radio-hls
mode stream
queue_bytes 1048576
argument -hide_banner
argument -loglevel
argument error
argument -probesize
argument 32
argument -analyzeduration
argument 0
argument -f
argument s16le
argument -ar
argument 48000
argument -ac
argument 2
argument -i
argument pipe:0
argument -c:a
argument aac
argument -b:a
argument 96k
argument -f
argument hls
argument -hls_time
argument 1
argument -hls_list_size
argument 3
argument -hls_flags
argument delete_segments
argument -hls_segment_filename
argument C:/radio-hls/radio_%03d.ts
argument -y
argument C:/radio-hls/radio.m3u8
```

The directory must exist and be writable by the Nodus process. The host must also declare `NODUS_EXEC_CORE_ROOTS` as required by `exec_row.h`. Use an output directory outside the core repositories. An Apache static route may serve this directory to many listeners; this row does not configure Apache or access control.

The probe arguments are significant. The installed FFmpeg waited until stdin EOF before it emitted HLS segments with its default probe. With `-probesize 32 -analyzeduration 0`, the focused test observed a playlist and transport stream segments while the child remained open.

The row never waits for FFmpeg's stdin. `queue_bytes` caps bytes queued or currently being written. When a tick exceeds the limit, it leaves the original tensor handles on the stack and marks the tick unsuccessful. The row logs a child exit and starts another child when later input arrives. Row shutdown kills the child through `exec_pipe`'s job object. FFmpeg's stdout is drained and discarded for this output-to-files configuration.

The present Radio PCM RTP exit pulls `PcmBlock` from the headless graph and sends network-order L16 in RTP over UDP. It does **not** presently yield little-endian U8 tensors to this row. A graph adapter or an existing receiver must convert that source to the declared `s16le` input format before wiring the Radio feed to `exec.radio.hls`. This change proves the continuous process and encoder output path; it does not claim the Radio-to-HLS graph edge or browser playback is connected.

Tested with `tools/exec_row/exec_row_test.cpp` and the installed FFmpeg, using several PCM tensors on one open child process. Existing compressor span behavior remains covered by the same test.

## Stack-connected Radio follow-up

`nodus_radio` now accepts a final `hls=1` argument after `wait-for-start`. When the host declares `exec.radio.hls`, its one Free table uses these rows in order:

```text
radio.file_output -> radio.pcm_s16le -> radio.udp_broadcast -> radio.pcm_rtp -> exec.radio.hls
```

The source puts its actual float32 PCM block in the same stack frame as its spectrum data. `radio.pcm_s16le` converts and quantizes the samples into a U8 tensor, places that tensor beneath the spectrum pointer, and leaves the pointer on top for the existing UDP spectrum row. The RTP row still consumes its original canonical PCM FIFO. After the spectrum pointer is consumed, the declared exec row copies the tensor bytes to FFmpeg's open stdin. The conversion row destroys its owned tensor on the next tick, after the exec row has copied it. No second PCM FIFO or Python conversion path is involved.

The optional HLS mode requires the file source and PCM port. The `NODUS_EXEC_ROWS` and `NODUS_EXEC_CORE_ROOTS` environment variables must be set before `nodus_radio.exe` starts, because row registration occurs when the headless image loads. The normal Radio CLI and RTP route remain valid without the final HLS argument.

`tools/test_radio_hls_stack.py` generated a five-second 48 kHz WAV, ran the real Nodus canvas with SDL dummy audio, and checked that HLS appeared while the child was still running. FFprobe identified AAC, 48 kHz, two channels in a generated transport stream segment. The same run received 938 RTP packets and reported zero native PCM drops. The test does not configure Apache or browser playback.
