"""Silent integration: real WAV -> canvas FIFO -> RTP -> existing native udp_router BATCH.
Arguments: radio.exe scope.dll fft.dll approved.wav udp_router.exe
No PA output: SDL dummy driver explicitly selected; isolated router is loopback only.
"""
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

HEADER = struct.Struct('<BH4sHI')
ENTRY = struct.Struct('<4sHH')

def free_port(kind):
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def exact(s, count):
    data = bytearray()
    while len(data) < count:
        part = s.recv(count - len(data))
        if not part:
            raise EOFError('router disconnected')
        data.extend(part)
    return bytes(data)

def frame(s):
    op, port, ip, peer, size = HEADER.unpack(exact(s, HEADER.size))
    assert size <= 2*1024*1024
    return op, port, exact(s, size)

def main():
    radio, scope, fft, wav, router_exe = map(lambda x: str(Path(x).resolve()), sys.argv[1:])
    control, destination = free_port(socket.SOCK_STREAM), free_port(socket.SOCK_DGRAM)
    processes = []
    with tempfile.TemporaryDirectory(prefix='nodus-pcm-') as cwd, tempfile.TemporaryFile(mode='w+t') as log:
        try:
            router = subprocess.Popen([router_exe, '127.0.0.1', str(control)], stdout=log, stderr=log, cwd=cwd)
            processes.append(router)
            deadline = time.monotonic()+5
            while True:
                try:
                    client = socket.create_connection(('127.0.0.1',control), timeout=.5)
                    break
                except OSError:
                    assert time.monotonic()<deadline, 'router startup timeout'
                    time.sleep(.05)
            with client:
                client.settimeout(1)
                assert frame(client)[0]==6
                client.sendall(HEADER.pack(1,destination,b'\0'*4,0,0))
                assert frame(client)[:2]==(7,destination)
                env=dict(os.environ, SDL_AUDIODRIVER='dummy')
                producer=subprocess.Popen([radio,scope,fft,'127.0.0.1',str(free_port(socket.SOCK_DGRAM)),'3','5','1024',wav,'1',str(destination)],stdout=log,stderr=log,cwd=cwd,env=env)
                processes.append(producer)
                packets=[]; deadline=time.monotonic()+15
                while producer.poll() is None:
                    assert time.monotonic()<deadline, 'producer timeout'
                    try:
                        op, port, body=frame(client)
                    except socket.timeout:
                        continue
                    assert op==8 and port==destination,(op,port,body[:120])
                    count=struct.unpack_from('<I',body)[0]; at=4
                    for _ in range(count):
                        ip,peer,size=ENTRY.unpack_from(body,at); at+=ENTRY.size
                        packet=body[at:at+size];at+=size
                        assert size==1036
                        v,pt,seq,stamp,ssrc=struct.unpack_from('>BBHII',packet)
                        assert (v,pt)==(128,96)
                        if packets:
                            assert seq==(packets[-1][0]+1)%65536
                            assert stamp==(packets[-1][1]+256)%2**32
                            assert ssrc==packets[-1][2]
                        pcm=struct.unpack_from('>512h',packet,12)
                        assert pcm[::2]==pcm[1::2], 'mono source should duplicate into stereo'
                        packets.append((seq,stamp,ssrc,max(map(abs,pcm))))
                    assert at==len(body)
                producer.wait(timeout=5)
                assert producer.returncode==0,producer.returncode
                assert len(packets)>300,len(packets)
                assert max(x[3] for x in packets)>100
                print(f'PASS {len(packets)} real RTP L16 packets through native router BATCH; sequence/timestamps/stereo/PCM verified; silent SDL dummy')
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate();process.wait(timeout=5)
            log.seek(0)
            lines=log.read().splitlines()
            print('\n'.join(x for x in lines if x.startswith(('Radio:', 'radio_', 'capture=', 'shutdown_'))))

if __name__=='__main__':main()
