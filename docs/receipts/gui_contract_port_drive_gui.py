import os, sys, time, subprocess, socket, pathlib, re, json
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(r"C:/dev/Powershell/nodus"); HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT/"tools"))
from contract_service_peer import EnginePeer, ServicePeer, connect_loopback, ROLE_ENGINE
exe = ROOT/"build-bridging/Release/frontend_shell.exe"
canvas = HERE/"messenger.gui.canvas"
cmd = [str(exe), f"--workspace={canvas}", "--acquire=gui.proof:4", "--width=640", "--height=400"]
print("CMD", " ".join(cmd))
so = open(HERE/"gui_stdout.txt","w"); se = open(HERE/"gui_stderr.txt","w")
t0 = time.monotonic()
proc = subprocess.Popen(cmd, stdout=so, stderr=se, cwd=str(ROOT))
print("PID", proc.pid)
port = None; line = None
try:
    while time.monotonic()-t0 < 30:
        if proc.poll() is not None: print("GUI exited early rc", proc.returncode); break
        so.flush()
        m = re.search(r"contract_port gui\.proof 127\.0\.0\.1:(\d+)", (HERE/"gui_stdout.txt").read_text(errors="replace"))
        if m: port = int(m.group(1)); line = m.group(0); break
        time.sleep(0.1)
    if port:
        t_line = time.monotonic()-t0
        print(f"LINE '{line}' at +{t_line:.2f}s")
        ev = {"admitted":[], "routed":[], "receipts":[]}; ts = {}
        def mark(k):
            ts.setdefault(k, time.monotonic()-t0)
        class Producer:
            def admitted(self, i): ev["admitted"].append(i); mark("admitted")
            def route(self, r): ev["routed"].append(r); mark("route"); return 1
        engine = EnginePeer(connect_loopback(port, ROLE_ENGINE, 7), Producer()); mark("engine_connected")
        conn = socket.create_connection(("127.0.0.1", port), timeout=5.0); mark("host_connected")
        with ThreadPoolExecutor(max_workers=1) as ex:
            host = ServicePeer(conn, ex, lambda loan: 555,
                               lambda loan, ok: (ev["receipts"].append(ok), mark("receipt")),
                               announce=True, identity=1)
            ok = engine.submit((100,7,3,4)); mark("submitted"); print("submit", ok)
            dl = time.monotonic()+20
            while time.monotonic() < dl:
                engine.poll(); host.poll()
                if ev["admitted"] and ev["routed"] and ev["receipts"]: break
                time.sleep(0.002)
        print("events", json.dumps({k:(v if k!="routed" else [list(map(int,r)) if hasattr(r,'__iter__') else r for r in v]) for k,v in ev.items()}, default=str))
        print("timings_s", json.dumps({k:round(v,3) for k,v in ts.items()}))
        print("errors", engine.error, host.error)
        try: engine.close(); host.close()
        except Exception as e: print("close", e)
finally:
    time.sleep(1)
    print("alive before kill:", proc.poll() is None)
    out = subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],capture_output=True,text=True)
    print("taskkill:", out.stdout.strip(), out.stderr.strip())
    time.sleep(1)
    chk = subprocess.run(["tasklist","/FI",f"PID eq {proc.pid}"],capture_output=True,text=True).stdout
    print("gone:", str(proc.pid) not in chk, "rc", proc.poll())
    so.close(); se.close()
