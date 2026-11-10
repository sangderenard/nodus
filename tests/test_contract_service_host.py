import os,sys,socket,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
# Dependency directories for the existing qualified supporting libraries.
keepers=[os.add_dll_directory(str(ROOT/'build/Release')),
 os.add_dll_directory(r'C:\Users\alber\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib')]
from nodus_mail_host import NativeMailHost
from contract_service_peer import ServicePeer
os.chdir(ROOT/'build/contract-loopback-isolated')
a,b=socket.socketpair()
admitted=[]; results=[]; rejected=[]
def request():return None if admitted else (77,1,9,123)
def route(r):
    if not rejected:
        rejected.append(r.loan.generation);return -1
    assert r.loan.generation!=rejected[0]
    assert r.loan.work.id==77 and r.payload==456
    results.append(r.loan.generation);return 1
host=NativeMailHost(ROOT/'build/Release'/os.environ.get('NODUS_TEST_HOST_DLL','nodus_contract_service.dll'),a,1,request,route,admitted.append)
with ThreadPoolExecutor(max_workers=1) as executor:
    receipts=[]
    peer=ServicePeer(b,executor,lambda loan:456,lambda loan,ok:receipts.append(ok))
    for _ in range(20000):
        host.step();peer.poll()
        if results and len(receipts)>=2:break
        time.sleep(.0001)
    assert results and rejected and len(receipts)>=2,(results,rejected,receipts,peer.error,host.errors)
    assert not host.errors and peer.error is None
    assert host.retire(77)
    peer.close()
host.close()
print('Native two-table graph -> Python pure service -> native FIFO -> reject/reoffer -> owner commit: PASS')
# Deferred server double-check runs only on completed client returns, outside
# native route callbacks. One check per returned result, including rejection.
a,b=socket.socketpair()
admitted=[]; pending={}; decisions={}; client_calls=[]; checks=[]; commits=[]
clock=[0]
def request_delayed():return None if admitted else (88,1,10,124)
def route_delayed(r):
    generation=r.loan.generation
    if generation not in pending:pending[generation]=(clock[0]+5,r)
    decision=decisions.get(generation,0)
    if decision==1:commits.append(generation)
    return decision
host=NativeMailHost(ROOT/'build/Release'/os.environ.get('NODUS_TEST_HOST_DLL','nodus_contract_service.dll'),a,1,request_delayed,route_delayed,admitted.append)
def participant(loan):
    client_calls.append(loan.generation)
    return 999 if len(client_calls)==1 else 456
with ThreadPoolExecutor(max_workers=1) as executor:
    receipts=[]
    peer=ServicePeer(b,executor,participant,lambda loan,ok:receipts.append(ok))
    # No participant polling means no completed return and absolutely no check.
    for _ in range(8):host.step()
    assert not checks and not pending
    for tick in range(20000):
        clock[0]=tick
        host.step();peer.poll()
        for generation,(due,result) in list(pending.items()):
            if generation in decisions or tick<due:continue
            assert generation in client_calls
            checks.append(generation)
            decisions[generation]=1 if result.payload==456 else -1
        if commits and len(receipts)>=2:break
        time.sleep(.0001)
    assert len(client_calls)==2 and len(checks)==2 and len(commits)==1
    assert len(set(checks))==2 and not host.errors and peer.error is None
    assert host.retire(88)
    peer.close()
host.close()
print('Deferred validation: two participant returns, exactly two server checks, one committed result; no offer-time check: PASS')

