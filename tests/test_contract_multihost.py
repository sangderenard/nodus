"""One retained parent batch, disjoint native contracts, two ready participants."""
import os,sys,socket,time,collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from nodus_mail_host import NativeMailHost
from contract_service_peer import ServicePeer
keepers=[os.add_dll_directory(str(ROOT/'build/Release')),os.add_dll_directory(str(Path(sys.executable).parent/'Lib/site-packages/torch/lib'))]
os.chdir(ROOT/'build/contract-loopback-isolated')
# One frozen parent lease, 512 samples, eight nonoverlapping ranges.
chunks={i+1:('parent-lease-v1',i*64,64) for i in range(8)}
admitted=set();returned={};computed=[];receipts=[]
def request():
 for identifier in chunks:
  if identifier not in admitted:return (identifier,7,13,identifier)
 return None
def route(result):
 identifier=result.loan.work.id
 assert result.loan.work.payload==identifier and result.loan.work.revision==13
 assert identifier not in returned
 assert result.payload==identifier
 returned[identifier]=result.loan.host
 return 1
s1,p1=socket.socketpair();s2,p2=socket.socketpair()
host=NativeMailHost(ROOT/'build/Release/nodus_contract_service_multihost.dll',s1,7,request,route,admitted.add,capacity=8,ticks=10000)
host.add_participant(s2,2)
def compute(loan):
 assert loan.payload in chunks
 computed.append((loan.contract,loan.host,loan.generation))
 time.sleep(.002)
 return loan.payload
with ThreadPoolExecutor(max_workers=2) as executor:
 peers=[ServicePeer(p,executor,compute,lambda loan,ok:receipts.append((loan.contract,ok))) for p in (p1,p2)]
 for _ in range(20000):
  host.step()
  for peer in peers:peer.poll()
  if len(returned)==8 and len(receipts)==8:break
  time.sleep(.0001)
 assert len(returned)==8 and len(computed)==8,(returned,computed,receipts)
 assert len({entry[0] for entry in computed})==8
 assert all(peer.error is None for peer in peers) and not host.errors
 assert set(returned.values())=={1,2}
 covered=[n for key in returned for n in range(chunks[key][1],chunks[key][1]+chunks[key][2])]
 assert sorted(covered)==list(range(512))
 for key in returned:assert host.retire(key)
 for peer in peers:peer.close()
host.close()
print('Native entire-batch allocation PASS:',dict(collections.Counter(returned.values())), '512 samples exactly once; eight disjoint chunks; two participants')
# Expiry recovery retains the exact chunk and changes only native generation/host.
s1,p1=socket.socketpair();s2,p2=socket.socketpair();p1.setblocking(False)
admitted=set();commits=[];old=[];fresh=[];wire=bytearray()
from contract_service_peer import encode,decode,OFFER,ACCEPT,RESULT
host=NativeMailHost(ROOT/'build/Release/nodus_contract_service_multihost.dll',s1,7,
 lambda:None if admitted else (99,7,13,99),lambda r:commits.append((r.loan.generation,r.loan.host)) or 1,admitted.add,capacity=1,ticks=12)
host.step();host.step()
host.add_participant(s2,2)
with ThreadPoolExecutor(max_workers=1) as executor:
 peer=ServicePeer(p2,executor,lambda loan:fresh.append(loan) or 99,lambda *_:None)
 for tick in range(1000):
  host.step();peer.poll()
  try:wire.extend(p1.recv(72-len(wire)))
  except BlockingIOError:pass
  if len(wire)==72:
   kind,loan,payload=decode(wire);wire.clear()
   if kind==OFFER and not old:
    old.append(loan);p1.sendall(encode(ACCEPT,loan))
  if commits:break
  time.sleep(.001)
 assert len(old)==1 and len(fresh)==1 and len(commits)==1,(old,fresh,commits)
 assert old[0].contract==fresh[0].contract==99 and old[0].payload==fresh[0].payload==99
 assert old[0].generation!=fresh[0].generation and fresh[0].host==2
 p1.sendall(encode(RESULT,old[0],99))
 for _ in range(8):host.step();peer.poll()
 assert len(commits)==1 and not host.errors
 assert host.retire(99)
 peer.close()
p1.close();host.close()
print('Native expiry recovery PASS: same chunk reoffered to another host, fresh generation, late stale result did not commit')



