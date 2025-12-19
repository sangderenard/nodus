Scaffold for ThreadManager and Scheduler

Files added:
- include/thread_manager.h
- src/thread_manager.cpp
- include/kpn_scheduler.h
- src/kpn_scheduler.cpp
- tests/scaffold_test.cpp

Next steps:
- Implement real reader-table allocation and per-edge min-reader snapshots in `ThreadManager::Impl`.
- Port DFS/topological scheduler from `ilpscheduler.py` into `KpnScheduler` and wire into the tick path.
- Add CMake targets for the new sources and unit tests.
- Integrate reader registration into `gp_table_edge_subscribe` so readers are claimed before publish in Scheduled mode.
