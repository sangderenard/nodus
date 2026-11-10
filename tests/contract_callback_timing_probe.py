"""Isolated callback wait attribution. No server, GL or production DLL change."""
import ctypes as C,json,os,socket,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from nodus_mail_host import NativeMailHost
keepers=[os.add_dll_directory(str(ROOT/'build/Release')),os.add_dll_directory(str(Path(sys.executable).parent/'Lib/site-packages/torch/lib'))]
os.chdir(ROOT/'build/contract-loopback-isolated')

def run(busy,interval):
 stop=threading.Event()
 def competing_python():
  number=0
  while not stop.is_set():number=(number+1)%1234567
 old=sys.getswitchinterval();sys.setswitchinterval(interval)
 a,b=socket.socketpair();host=NativeMailHost(ROOT/'build/Release/nodus_contract_service_diagnostic.dll',a,1,lambda:None,lambda _:0,lambda _:None)
 thread=threading.Thread(target=competing_python) if busy else None
 if thread:thread.start()
 try:
  start=time.perf_counter();cpu=time.thread_time()
  for _ in range(100):host.step()
  wall=time.perf_counter()-start;cpu=time.thread_time()-cpu
  values=(C.c_double*6)()
  host.lib.nodus_mail_timings.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_size_t]
  assert host.lib.nodus_mail_timings(host.handle,values,6)
  return dict(busy_python_thread=busy,switch_interval=interval,python_call_wall=wall,calling_thread_cpu=cpu,native_step=values[0],request_callback=values[1],route_callback=values[2],admitted_callback=values[3],socket_exchange=values[4],steps=int(values[5]),native_residual=values[0]-sum(values[1:5]))
 finally:
  stop.set()
  if thread:thread.join()
  host.close();b.close();sys.setswitchinterval(old)
results=[run(False,.005),run(True,.005),run(True,.001)]
print(json.dumps(results,indent=2))
(ROOT/'build/contract-callback-timings.json').write_text(json.dumps(results,indent=2))
