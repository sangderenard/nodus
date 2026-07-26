#include "graph_runtime.h"

GraphRuntime::GraphRuntime() {
    thread_mgr = std::make_unique<ThreadManager>();
    thread_mgr->set_graph_runtime(this);
}

GraphRuntime::~GraphRuntime() = default;
