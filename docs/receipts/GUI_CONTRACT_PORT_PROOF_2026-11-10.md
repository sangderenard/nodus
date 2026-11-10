# GUI contract-port proof (HEAD 2ed65f2)
Env: NODUS_HEADLESS_TEST_DLL=C:/dev/Powershell/nodus/build-bridging/Release/nodus_headless.dll; PATH prepended with build-bridging/Release and torch/lib; system python.
Commands (cwd out/gui-contract-port-proof): `python make_canvas.py`; `cp messenger.canvas messenger.gui.canvas`; `python drive_gui.py`
GUI cmd: frontend_shell.exe --workspace=<abs>/messenger.gui.canvas --acquire=gui.proof:4 --width=640 --height=400
Saved trailer: `CONTRACT_PORT gui.proof 4 0 0 1 0 47611` (plus EDGE_FIFO 0 0 1 0 64 4 0 1397379633 2 8 0)
GUI printed: `contract_port gui.proof 127.0.0.1:47611` at +2.33 s after launch. PID 78652.
Protocol (s since launch): engine connected 2.344; host connected 2.360; submit (100,7,3,4) True 2.360; engine Admitted [100] 2.375; Route 2.422 (result [100,7,3,4,1,1,69,555]); host Receipt True 2.454. No peer errors.
Termination: GUI alive after 1 s; taskkill /PID 78652 /T /F SUCCESS; tasklist confirms gone.
Autosave: messenger.gui.canvas after run is byte-identical to messenger.canvas (375 bytes, cmp silent, both retain EDGE_FIFO + CONTRACT_PORT). Interval is 2.0 s (kCanvasAutosaveIntervalSeconds) and the run lasted ~6 s, yet the file is unchanged: no rewrite observed (likely not dirty; cause not investigated).
Not proven: autosave stripping the trailer (no autosave write occurred on an unedited canvas; a GUI edit was not exercised); rendering/visual correctness of the window; behaviour on GUI edit; clean (non-kill) shutdown.
