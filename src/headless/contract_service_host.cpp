// Bounded native host for Python development services. All calls and callbacks
// are serialized on the owning engine lane; socket IO itself is nonblocking.
#include "contract_socket.h"
#include "contract_negotiator_tool.h"
#include "nodus_headless_abi.h"
#include <memory>
#include <atomic>
#include <chrono>
#ifdef _WIN32
#define MAIL_API extern "C" __declspec(dllexport)
#else
#define MAIL_API extern "C"
#endif
using namespace nodus::contracts;
using Request=int(*)(void*,Contract*);
using Route=int(*)(void*,const Result*);
using Admitted=void(*)(void*,uint64_t);
using WallClock=std::chrono::steady_clock;
struct Span {double& total;WallClock::time_point start=WallClock::now(); explicit Span(double& t):total(t){} ~Span(){total+=std::chrono::duration<double>(WallClock::now()-start).count();}};
struct MailHost {
    // step, request callback, route callback, admitted callback, socket exchange, step count
    double timings[6]={};
    ContractNegotiator queue;
    ContractSocket socket;
    std::vector<std::unique_ptr<ContractSocket>> participants;
    NodusHeadlessGraph* graph=nullptr;
    std::string prefix; uint64_t rejected_id=0,rejected_generation=0;
    MailHost(uintptr_t fd,size_t capacity,uint64_t ticks):queue(capacity,
      [](const Peer&,const std::vector<ContractNegotiator::Entry>& entries){
        for(size_t i=0;i<entries.size();++i)if(entries[i].state==ContractNegotiator::State::Unaccepted)return i;
        return size_t(-1);
      },ticks,false),socket(SOCKET(fd)){}
    ~MailHost(){if(graph)nodus_headless_destroy(graph); if(!prefix.empty()){tool_registry_global().unregister_tool(prefix+".market");tool_registry_global().unregister_tool(prefix+".worker");}}
};
MAIL_API void* nodus_mail_create(uintptr_t fd,uint64_t engine,size_t capacity,uint64_t ticks,
 Request request,Route route,Admitted admitted,void* user){
    if(!request || !route || !admitted || !engine)return nullptr;
    try {
      auto host=std::make_unique<MailHost>(fd,capacity,ticks);auto* h=host.get();
      if(!h->queue.add_engine({engine,[=](const Result& r){Span span(h->timings[2]);int decision=route(user,&r); if(decision==-1){h->rejected_id=r.loan.work.id;h->rejected_generation=r.loan.generation;} return decision==1;},
        [=](Contract& c){Span span(h->timings[1]);return request(user,&c)!=0;},[=](uint64_t id){Span span(h->timings[3]);admitted(user,id);}}))return nullptr;
      if(!h->queue.add_host({1,[=](const Loan* l){Span span(h->timings[4]);return h->socket.exchange(l);},
        [=](uint64_t g,bool ok){h->socket.receipt(g,ok);}}))return nullptr;
      static std::atomic<uint64_t> serial{0};
      std::string prefix="service.mail."+std::to_string(++serial);
      h->prefix=prefix;
      if(!register_negotiator_tools(tool_registry_global(),h->queue,prefix))return nullptr;
      h->graph=nodus_headless_create();if(!h->graph)return nullptr;
      GP_CanvasModuleDesc desc{};desc.w=desc.h=64;
      int m=nodus_headless_add_module(h->graph,&desc),w=nodus_headless_add_module(h->graph,&desc);
      int mr=nodus_headless_bind_plugin_tool(h->graph,m,(prefix+".market").c_str());
      int wr=nodus_headless_bind_plugin_tool(h->graph,w,(prefix+".worker").c_str());
      if(m<0 || w<0 || mr<0 || wr<0)return nullptr;
      GP_CanvasEdgeDesc edge{};edge.a_module=m;edge.a_contact_idx=mr;edge.b_module=w;edge.b_contact_idx=wr;
      int id=nodus_headless_add_edge(h->graph,&edge,0);if(id<0)return nullptr;
      NodusHeadlessFifoConfig config{};config.token_bytes=sizeof(Result);config.slots=capacity;
      config.top_k=0;config.type_id=0x534A5231;config.layout=2;config.dtype=8;
      if(!nodus_headless_configure_edge_fifo(h->graph,id,&config))return nullptr;
      if(!h->queue.bind_result_transport([=](const Result& r){NodusHeadlessPublishResult p{};
        return nodus_headless_publish_edge_token(h->graph,id,&r,sizeof(r),&p) && p.accepted && !p.dropped;
      },[=](Result& r){size_t n=0;return nodus_headless_consume_edge_token(h->graph,id,&r,sizeof(r),&n) && n==sizeof(r);} ))return nullptr;
      if(!nodus_headless_set_scheduler_mode(h->graph,1))return nullptr;
      return host.release();
    }catch(...){return nullptr;}
}
MAIL_API int nodus_mail_step(void* host){try{if(!host)return 0;auto* h=static_cast<MailHost*>(host);Span span(h->timings[0]);++h->timings[5];int ok=nodus_headless_step(h->graph,1.0/60.0);if(h->rejected_id){h->queue.reject_pending(h->rejected_id,h->rejected_generation);h->rejected_id=0;}return ok;}catch(...){return 0;}}
MAIL_API int nodus_mail_retire(void* host,uint64_t id){return host && static_cast<MailHost*>(host)->queue.retire(id);}
MAIL_API void nodus_mail_destroy(void* host){delete static_cast<MailHost*>(host);}


MAIL_API int nodus_mail_timings(void* host,double* output,size_t count){if(!host || !output || count<6)return 0;auto* h=static_cast<MailHost*>(host);for(size_t i=0;i<6;++i)output[i]=h->timings[i];return 1;}

MAIL_API int nodus_mail_add_participant(void* host,uintptr_t fd,uint64_t participant){
    if(!host)return 0;
    try {auto* h=static_cast<MailHost*>(host);auto socket=std::make_unique<ContractSocket>(SOCKET(fd));auto* peer=socket.get();
      if(!h->queue.add_host({participant,[h,peer](const Loan* loan){Span span(h->timings[4]);return peer->exchange(loan);},
          [peer](uint64_t generation,bool accepted){peer->receipt(generation,accepted);}}))return 0;
      h->participants.push_back(std::move(socket));return 1;
    }catch(...){return 0;}
}
