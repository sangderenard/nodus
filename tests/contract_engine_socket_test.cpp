// Real loopback: ContractListener -> EngineSocket (Producer) + ContractSocket (host Peer)
// -> ContractNegotiator (standalone FIFO). No graph, no ThreadManager.
#include "contract_listener.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <chrono>
#include <vector>
using namespace nodus::contracts;
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)

static SOCKET connect_to(uint16_t port){
    SOCKET s=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);CHECK(s!=INVALID_SOCKET);
    sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);a.sin_port=htons(port);
    CHECK(connect(s,reinterpret_cast<sockaddr*>(&a),sizeof(a))==0);return s;
}
static WireMessage hello(uint64_t role,uint64_t id){WireMessage m{WireKind::Hello,{}};m.result.loan.work={role,id,2,0};return m;}
static WireMessage submit(Contract c){WireMessage m{WireKind::Submit,{}};m.result.loan.work=c;return m;}
static WireMessage decision(uint64_t id,uint64_t generation,uint64_t value){
    WireMessage m{WireKind::Decision,{}};m.result.loan.work.id=id;m.result.loan.generation=generation;m.result.payload=value;return m;
}
int main(){
    ensure_sockets();
    ContractNegotiator negotiator(4,[](const Peer&,const std::vector<ContractNegotiator::Entry>& entries){
        for(size_t i=0;i<entries.size();++i)if(entries[i].state==ContractNegotiator::State::Unaccepted)return i;
        return size_t(-1);
    },10000,true);
    ContractListener listener(0);CHECK(listener.port()!=0);
    std::puts("OK listener bound to loopback ephemeral port");

    ContractSocket engine_client(connect_to(listener.port())),host_client(connect_to(listener.port()));
    const uint64_t kEngine=7,kHost=1;
    CHECK(engine_client.send_message(hello(2,kEngine)));
    CHECK(engine_client.send_message(submit({100,kEngine,3,4})));
    CHECK(host_client.send_message(hello(1,kHost)));

    std::vector<std::unique_ptr<EngineSocket>> engines;std::vector<std::unique_ptr<ContractSocket>> hosts;
    int commits=0,rejects=0,hellos=0;uint64_t rej_id=0,rej_gen=0;bool reject_was_reofferable=false;
    auto on_hello=[&](SOCKET s,uint64_t role,uint64_t id){
        ++hellos;
        if(role==2){
            engines.push_back(std::make_unique<EngineSocket>(s));auto* e=engines.back().get();
            CHECK(negotiator.add_engine({id,[&,e](const Result& r){
                int d=e->route(r);
                if(d==-1){rej_id=r.loan.work.id;rej_gen=r.loan.generation;}
                if(d==1)++commits;
                return d==1;
            },[e](Contract& c){return e->request(c);},[e](uint64_t cid){e->admitted(cid);}}));
        }else{
            hosts.push_back(std::make_unique<ContractSocket>(s));auto* h=hosts.back().get();
            CHECK(negotiator.add_host({id,[h](const Loan* l){return h->exchange(l);},
                [h](uint64_t g,bool ok){h->receipt(g,ok);}}));
        }
    };

    bool admitted[2]={};int offers[2]={},routes[2]={},receipts=0;bool second_sent=false;
    const uint64_t kHostPayload=555;
    for(int tick=0;tick<5000 && !(commits==2 && receipts>=3 && rejects==1);++tick){
        listener.poll(on_hello);
        negotiator.market_turn();negotiator.worker_turn();
        if(rej_id){
            CHECK(negotiator.reject_pending(rej_id,rej_gen));++rejects;
            for(auto& e:negotiator.contracts())if(e.work.id==rej_id)reject_was_reofferable=e.state==ContractNegotiator::State::Unaccepted;
            rej_id=0;
        }
        engine_client.poll();host_client.poll();WireMessage m{};
        while(engine_client.receive(m)){
            const uint64_t id=m.result.loan.work.id;
            if(m.kind==WireKind::Admitted){CHECK(id==100 || id==101);admitted[id-100]=true;}
            if(m.kind==WireKind::Route){
                CHECK(id==100 || id==101);CHECK(m.result.payload==kHostPayload && m.result.loan.host==kHost && m.result.loan.work.engine==kEngine);
                int n=++routes[id-100];
                // contract 101 is rejected on its first route, committed on the re-offered one
                uint64_t verdict=(id==101 && n==1)?2:1;
                CHECK(engine_client.send_message(decision(id,m.result.loan.generation,verdict)));
            }
        }
        while(host_client.receive(m)){
            if(m.kind==WireKind::Offer){
                const uint64_t id=m.result.loan.work.id;CHECK(id==100 || id==101);++offers[id-100];
                CHECK(host_client.send_message({WireKind::Accept,{m.result.loan,0}}));
                CHECK(host_client.send_message({WireKind::Result,{m.result.loan,kHostPayload}}));
            }
            if(m.kind==WireKind::Receipt){CHECK(m.result.payload==1);++receipts;}
        }
        if(commits==1 && !second_sent){second_sent=true;CHECK(engine_client.send_message(submit({101,kEngine,5,6})));}
        engine_client.poll();host_client.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(hellos==2 && engines.size()==1 && hosts.size()==1);
    std::puts("OK listener validated Hello and handed engine/host sockets to adapters");
    CHECK(admitted[0] && admitted[1]);
    std::puts("OK engine client received Admitted for both submitted contracts");
    CHECK(offers[0]==1 && receipts>=1 && routes[0]==1);
    std::puts("OK host received Offer, replied Accept+Result, engine received Route, host received Receipt");
    CHECK(commits==2);
    for(auto& e:negotiator.contracts())CHECK(e.state==ContractNegotiator::State::Committed);
    CHECK(negotiator.contracts().size()==2);
    std::puts("OK Decision=1 committed both contracts in the negotiator");
    CHECK(rejects==1 && reject_was_reofferable && offers[1]==2 && routes[1]==2 && receipts==3);
    std::puts("OK Decision=2 rejected pending result; entry re-offered with a new generation and then committed");
    CHECK(!engine_client.closed() && !host_client.closed() && !engines[0]->closed() && !hosts[0]->closed());
    std::puts("loopback listener -> engine socket + host socket -> negotiator passed (no graph)");
    return 0;
}
