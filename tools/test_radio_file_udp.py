"""Bounded local NRAD acceptance test for the real native WAV source.
Usage: python tools/test_radio_file_udp.py <radio.exe> <scope.dll> <fft.dll> <wav>
SDL3 runtime must be on PATH. Plays three seconds of the approved WAV.
"""
import math
import socket
import struct
import subprocess
import sys
import time
import tempfile


def main():
    executable, scope, fft, wav = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="nodus-radio-test-") as workspace, tempfile.TemporaryFile(mode="w+t") as log, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(.2)
        process = subprocess.Popen([executable, scope, fft, "127.0.0.1", str(receiver.getsockname()[1]), "3", "5", "1024", wav, "1"], stdout=log, stderr=log, text=True, cwd=workspace)
        packets = []
        deadline = time.monotonic() + 15
        try:
            while process.poll() is None:
                assert time.monotonic() < deadline, "native radio exceeded test deadline"
                try:
                    data, peer = receiver.recvfrom(2048)
                except socket.timeout:
                    continue
                assert len(data) == 48, len(data)
                magic, version, bands, rate, index, rms = struct.unpack_from("<IIIIQf", data)
                power = struct.unpack_from("<5f", data, 28)
                assert (magic, version, bands, rate) == (0x4441524e, 1, 5, 48000)
                assert math.isfinite(rms) and all(math.isfinite(x) and 0 <= x <= 1 for x in power)
                assert not packets or index > packets[-1][0]
                packets.append((index, rms, peer[1]))
            process.wait(timeout=5)
            log.seek(0)
            stdout = log.read()
            print("\n".join(line for line in stdout.splitlines() if not line.startswith("DEBUG:")))
            assert process.returncode == 0, process.returncode
            assert len(packets) > 20, len(packets)
            assert max(x[1] for x in packets) > .001
            assert len({x[2] for x in packets}) == 1
            assert f"radio_udp_source_port={packets[0][2]}" in stdout
            print(f"PASS {len(packets)} real WAV NRAD frames, monotonic indices, finite nonzero RMS, matching source port")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
