#include "contract_socket.h"
#include "contract_negotiator_tool.h"
#include "nodus_headless_abi.h"
#include <cassert>
#include <cstdio>
#include <thread>
#include <chrono>
using namespace nodus::contracts;
int main(){
#ifdef _WIN32
    WSADATA data{};assert(WSAStartup(MAKEWORD(2,2),&data)==0);
#endif
    SOCKET listener=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);assert(listener!=INVALID_SOCKET);
    sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);address.sin_port=0;
    assert(bind(listener,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0);assert(listen(listener,1)==0);
#ifdef _WIN32
    int length=sizeof(address);
#else
    socklen_t length=sizeof(address);
#endif
    assert(getsockname(listener,reinterpret_cast<sockaddr*>(&address),&length)==0);
    SOCKET client=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);assert(client!=INVALID_SOCKET);
    assert(connect(client,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0);
    SOCKET accepted=accept(listener,nullptr,nullptr);assert(accepted!=INVALID_SOCKET);
#ifdef _WIN32
    closesocket(listener);
#else
    close(listener);
#endif
    ContractSocket server(accepted),remote(client);
    int committed=0,receipts=0; int per_engine[3]={}; bool admitted[3]={};
    ContractNegotiator negotiator(4,[](const Peer&,const std::vector<ContractNegotiator::Entry>& entries){
        for(size_t i=0;i<entries.size();++i)if(entries[i].state==ContractNegotiator::State::Unaccepted)return i;
        return size_t(-1);
    },10000,
#ifdef CONTRACT_SOCKET_GRAPH_TEST
    false
#else
    true
#endif
    );
    for(uint64_t id=1;id<=2;++id){
        assert(negotiator.add_engine({id,[&,id](const Result& r){
            assert(r.loan.work.engine==id && r.loan.work.revision==id+7);
            assert(r.payload==id*1234);++per_engine[id];++committed;return true;
        },[&,id](Contract& work){
            if(admitted[id])return false;
            work={id,id,id+7,id+8};return true;
        },[&,id](uint64_t contract){assert(contract==id);admitted[id]=true;}}));
    }
    assert(negotiator.add_host({1,[&](const Loan* offer){return server.exchange(offer);},
        [&](uint64_t generation,bool ok){assert(server.receipt(generation,ok));}}));

    assert(register_negotiator_tools(tool_registry_global(),negotiator,"socket.proof"));
#ifdef CONTRACT_SOCKET_GRAPH_TEST
    auto* graph=nodus_headless_create();assert(graph);
    GP_CanvasModuleDesc desc{};desc.w=desc.h=64;
    int market=nodus_headless_add_module(graph,&desc),worker=nodus_headless_add_module(graph,&desc);
    assert(market>=0 && worker>=0);
    int mr=nodus_headless_bind_plugin_tool(graph,market,"socket.proof.market");
    int wr=nodus_headless_bind_plugin_tool(graph,worker,"socket.proof.worker");assert(mr>=0 && wr>=0);
    GP_CanvasEdgeDesc edge{};edge.a_module=market;edge.a_contact_idx=mr;edge.b_module=worker;edge.b_contact_idx=wr;
    int edge_id=nodus_headless_add_edge(graph,&edge,0);assert(edge_id>=0);
    NodusHeadlessFifoConfig config{};config.token_bytes=sizeof(Result);config.slots=4;
    config.top_k=0;config.type_id=0x534A5231;config.layout=2;config.dtype=8;
    assert(nodus_headless_configure_edge_fifo(graph,edge_id,&config));
    assert(negotiator.bind_result_transport([&](const Result& result){
        NodusHeadlessPublishResult decision{};
        assert(nodus_headless_publish_edge_token(graph,edge_id,&result,sizeof(result),&decision));
        assert(!decision.dropped);return decision.accepted!=0;
    },[&](Result& result){size_t written=0;
        bool consumed=nodus_headless_consume_edge_token(graph,edge_id,&result,sizeof(result),&written)!=0;
        assert(!consumed || written==sizeof(result));return consumed;
    }));
    assert(nodus_headless_set_scheduler_mode(graph,1));
#else
    auto market_tool=tool_registry_global().create("socket.proof.market");
    auto worker_tool=tool_registry_global().create("socket.proof.worker");
    assert(market_tool && worker_tool);
    ToolStackContext context{};
#endif
    for(int tick=0;tick<2000 && (committed<2 || receipts<2);++tick){
#ifdef CONTRACT_SOCKET_GRAPH_TEST
        assert(nodus_headless_step(graph,1.0/60.0));
#else
        market_tool->execute_stack(context);worker_tool->execute_stack(context);assert(context.succeeded);
#endif
        remote.poll();WireMessage m{};
        while(remote.receive(m)){
            if(m.kind==WireKind::Offer){assert(remote.send_message({WireKind::Accept,{m.result.loan,0}}));
                assert(remote.send_message({WireKind::Result,{m.result.loan,m.result.loan.work.engine*1234}}));}
            if(m.kind==WireKind::Receipt){assert(m.result.payload==1);++receipts;}
        }
        remote.poll();std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(committed==2 && receipts==2 && per_engine[1]==1 && per_engine[2]==1);assert(!server.closed() && !remote.closed());
#ifdef CONTRACT_SOCKET_GRAPH_TEST
    nodus_headless_destroy(graph);
    std::puts("real headless ThreadManager -> loopback socket -> Nodus FIFO -> producer commit passed");
#else
    std::puts("native tools -> loopback socket -> Nodus FIFO -> producer commit passed (no scheduler)");
#endif
}
