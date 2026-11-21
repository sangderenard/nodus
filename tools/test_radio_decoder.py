"""Real silent radio decoder integration. Args: radio.exe scope.dll fft.dll ffmpeg.exe scratch-dir.
Generated local audio only; isolated UDP sockets, SDL dummy, no live service.
"""
import os, pathlib, socket, struct, subprocess, sys, time
radio,scope,fft,ffmpeg,scratch=sys.argv[1:]
scratch=pathlib.Path(scratch).resolve();scratch.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,SDL_AUDIODRIVER='dummy',NODUS_RADIO_DECODER=ffmpeg)
def run(name,seconds,loop):
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as pcm, socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as spectrum:
        pcm.bind(('127.0.0.1',0));pcm.settimeout(.05);spectrum.bind(('127.0.0.1',0))
        with open(scratch/'radio-test.log','w+') as log:
            p=subprocess.Popen([radio,scope,fft,'127.0.0.1',str(spectrum.getsockname()[1]),str(seconds),'5','1024',str(scratch/name),str(loop),str(pcm.getsockname()[1])],env=env,cwd=scratch,stdout=log,stderr=log)
            packets=[];bye=False;deadline=time.monotonic()+12
            try:
                while p.poll() is None:
                    assert time.monotonic()<deadline,'timeout'
                    try: data=pcm.recv(65536)
                    except socket.timeout: continue
                    if len(data)==52 and data[1]==200 and data[45]==203: bye=True;continue
                    assert data[:2]==b'\x80\x60'
                    _,_,seq,stamp,ssrc=struct.unpack('>BBHII',data[:12])
                    frames=(len(data)-12)//4
                    if packets:
                        prior=packets[-1];assert seq==(prior[0]+1)%65536;assert stamp==(prior[1]+prior[2])%2**32
                    samples=struct.unpack('>'+str(frames*2)+'h',data[12:]);assert samples[::2]==samples[1::2]
                    packets.append((seq,stamp,frames,max(map(abs,samples))))
                assert p.returncode==0,p.returncode
                assert packets and max(x[3] for x in packets)>100
                if not loop and not seconds: assert sum(x[2] for x in packets)==14400,(len(packets),sum(x[2] for x in packets));assert bye
                if seconds: assert len(packets)>250,len(packets)
                print('PASS',name,'loop',loop,'frames',sum(x[2] for x in packets),'BYE',bye)
            finally:
                if p.poll() is None:p.terminate();p.wait(timeout=5)
                log.seek(0);print('\n'.join(x for x in log.read().splitlines() if x.startswith(('Radio','capture=','pcm_','shutdown_'))))
for name,duration in [('short.wav',.3),('long.wav',180)]:
    subprocess.run([ffmpeg,'-nostdin','-v','error','-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t',str(duration),'-ac','2','-c:a','pcm_s16le','-y',str(scratch/name)],check=True)
run('short.wav',0,0)
run('short.wav',3,1)
run('long.wav',3,0)


subprocess.run([ffmpeg,'-nostdin','-v','error','-i',str(scratch/'short.wav'),'-c:a','libmp3lame','-y',str(scratch/'short.mp3')],check=True)
run('short.mp3',0,0)
(scratch/'invalid.audio').write_bytes(b'not an audio file')
failed=subprocess.run([radio,scope,fft,'127.0.0.1','9','0','5','1024',str(scratch/'invalid.audio'),'0'],env=env,cwd=scratch,capture_output=True,text=True,timeout=10)
assert failed.returncode!=0
print('PASS invalid decoder input refuses without hanging')
