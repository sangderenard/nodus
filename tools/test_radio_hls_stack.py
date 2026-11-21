"""Real Radio stack conversion -> open exec row -> FFmpeg HLS, with RTP retained.

Usage: python tools/test_radio_hls_stack.py <nodus_radio.exe> <scope.dll> <fft.dll> <ffmpeg.exe>
Uses SDL's dummy audio driver and an isolated loopback receiver; opens no speaker.
"""
import math
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import wave


def port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    radio, scope, fft, ffmpeg = [Path(arg).resolve(strict=True) for arg in sys.argv[1:5]]
    with tempfile.TemporaryDirectory(prefix='nodus-radio-hls-') as directory:
        root = Path(directory)
        source, playlist, declaration = root / 'source.wav', root / 'radio.m3u8', root / 'exec_rows.txt'
        with wave.open(str(source), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            sample = struct.Struct('<h')
            for frame in range(48000 * 5):
                wav.writeframesraw(sample.pack(int(12000 * math.sin(2 * math.pi * 440 * frame / 48000))))
        arguments = ['-hide_banner', '-loglevel', 'error', '-probesize', '32',
                     '-analyzeduration', '0', '-f', 's16le', '-ar', '48000', '-ac', '2',
                     '-i', 'pipe:0', '-c:a', 'aac', '-b:a', '96k', '-f', 'hls',
                     '-hls_time', '1', '-hls_list_size', '3', '-hls_flags', 'delete_segments',
                     '-hls_segment_filename', str(root / 'radio_%03d.ts'), '-y', str(playlist)]
        declaration.write_text('\n'.join(['row exec.radio.hls', 'name Radio AAC HLS',
            f'program {ffmpeg}', f'directory {root}', 'mode stream', 'queue_bytes 1048576',
            *(f'argument {argument}' for argument in arguments)]) + '\n', encoding='utf-8')
        env = dict(os.environ, SDL_AUDIODRIVER='dummy', NODUS_EXEC_ROWS=str(declaration),
                   NODUS_EXEC_CORE_ROOTS=str(radio.parents[2]))
        env['PATH'] = os.pathsep.join([str(radio.parent), str(scope.parent), str(fft.parent),
            str(Path('C:/dev/Powershell/nodus/build-bridging/vcpkg_installed/x64-windows/bin')),
            str(Path(sys.executable).parent / 'Lib/site-packages/torch/lib'), env.get('PATH', '')])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(('127.0.0.1', 0))
            receiver.settimeout(.1)
            pcm_port = receiver.getsockname()[1]
            with (root / 'radio.log').open('w+', encoding='utf-8') as log:
                process = subprocess.Popen([str(radio), str(scope), str(fft), '127.0.0.1',
                    str(port()), '0', '5', '1024', str(source), '0', str(pcm_port), '0', '1'],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                published_while_running = False
                rtp = 0
                try:
                    deadline = time.monotonic() + 20
                    while process.poll() is None and time.monotonic() < deadline:
                        if playlist.exists() and '.ts' in playlist.read_text(errors='replace'):
                            published_while_running = True
                        try:
                            packet, _ = receiver.recvfrom(2048)
                        except socket.timeout:
                            continue
                        if len(packet) >= 12 and packet[0] == 0x80 and packet[1] == 96:
                            rtp += 1
                    if process.poll() is None:
                        raise AssertionError('Radio source did not finish')
                    assert process.returncode == 0, f'Radio exited {process.returncode}'
                    assert published_while_running, 'HLS playlist was not published with the pipe open'
                    assert rtp > 100, f'RTP route emitted only {rtp} packets'
                    segments = list(root.glob('radio_*.ts'))
                    assert segments, 'HLS listed no transport stream segments'
                    probe = subprocess.run([str(ffmpeg.with_name('ffprobe.exe')), '-v', 'error',
                        '-show_entries', 'stream=codec_name,sample_rate,channels', '-of', 'json',
                        str(segments[-1])], capture_output=True, text=True, check=True)
                    streams = json.loads(probe.stdout)['streams']
                    assert any(stream.get('codec_name') == 'aac' and stream.get('sample_rate') == '48000'
                               and stream.get('channels') == 2 for stream in streams), streams
                    print(f'PASS HLS published on open exec pipe; {rtp} RTP packets retained')
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
                    log.seek(0)
                    print('\n'.join(log.read().splitlines()[-12:]))


if __name__ == '__main__':
    main()
