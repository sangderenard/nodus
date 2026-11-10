#pragma once
// Loopback-only accept loop for contract peers. NO authentication: the caller
// authorizes by relying on the 127.0.0.1 bind and by the (role,id) it chooses
// to accept inside on_hello (same stance as contract_socket.h). Each accepted
// connection must send one 72-byte Hello frame (kind 5, v0=role 1 host/2 engine,
// v1=participant id, v2=protocol version 2); exactly those 72 bytes are
// consumed here, the rest of the stream belongs to the adapter the callee builds.
// Nonblocking, bounded work per poll(), never blocks a graph tick.
#include "contract_socket.h"
#include <functional>
#include <vector>
#ifdef _WIN32
#include <mutex>
#endif
namespace nodus::contracts {
// Process-wide WSAStartup once (no other header in include/ does this). No-op off Windows.
inline void ensure_sockets(){
#ifdef _WIN32
    static std::once_flag once;
    std::call_once(once,[]{WSADATA d{};if(WSAStartup(MAKEWORD(2,2),&d)!=0)throw std::runtime_error("WSAStartup failed");});
#endif
}
class ContractListener {
public:
    static constexpr size_t kAcceptPerPoll=8, kMaxPending=32;
    static constexpr unsigned kMaxPollsWithoutHello=2000;
    explicit ContractListener(uint16_t port){
        ensure_sockets();
        listener_=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
        if(listener_==INVALID_SOCKET)throw std::runtime_error("listener socket refused");
        sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);a.sin_port=htons(port);
        if(bind(listener_,reinterpret_cast<sockaddr*>(&a),sizeof(a))!=0 || listen(listener_,16)!=0 || !WireSocket::set_nonblocking(listener_)){
            WireSocket::close_socket(listener_);throw std::runtime_error("listener bind/listen refused");
        }
#ifdef _WIN32
        int len=sizeof(a);
#else
        socklen_t len=sizeof(a);
#endif
        if(getsockname(listener_,reinterpret_cast<sockaddr*>(&a),&len)!=0){WireSocket::close_socket(listener_);throw std::runtime_error("getsockname failed");}
        port_=ntohs(a.sin_port);
    }
    ~ContractListener(){
        for(auto& p:pending_)WireSocket::close_socket(p.s);
        WireSocket::close_socket(listener_);
    }
    ContractListener(const ContractListener&)=delete;
    ContractListener& operator=(const ContractListener&)=delete;
    uint16_t port() const{return port_;}
    // on_hello owns the socket it is given.
    void poll(std::function<void(SOCKET,uint64_t role,uint64_t id)> on_hello){
        for(size_t i=0;i<kAcceptPerPoll;++i){
            SOCKET c=accept(listener_,nullptr,nullptr);
            if(c==INVALID_SOCKET)break;
            if(pending_.size()>=kMaxPending || !WireSocket::set_nonblocking(c)){WireSocket::close_socket(c);continue;}
            pending_.push_back({c});
        }
        for(size_t i=0;i<pending_.size();){
            auto& p=pending_[i];bool drop=false,done=false;
            int n=::recv(p.s,reinterpret_cast<char*>(p.buf.data()+p.count),int(p.buf.size()-p.count),0);
            if(n>0)p.count+=size_t(n);
            else if(n==0 || !WireSocket::would_block())drop=true;
            if(!drop && p.count==p.buf.size()){
                WireMessage m{};
                const auto& w=m.result.loan.work;
                if(WireSocket::decode(p.buf,m) && m.kind==WireKind::Hello && w.revision==2 &&
                   (w.id==1 || w.id==2) && w.engine!=0)done=true;
                else drop=true;
            }else if(!drop && ++p.polls>kMaxPollsWithoutHello)drop=true;
            if(done){
                WireMessage m{};WireSocket::decode(p.buf,m);
                SOCKET s=p.s;uint64_t role=m.result.loan.work.id,id=m.result.loan.work.engine;
                pending_.erase(pending_.begin()+i);
                if(on_hello)on_hello(s,role,id);else WireSocket::close_socket(s);
            }else if(drop){WireSocket::close_socket(p.s);pending_.erase(pending_.begin()+i);}
            else ++i;
        }
    }
private:
    struct Pending{SOCKET s;WireSocket::Frame buf{};size_t count=0;unsigned polls=0;};
    SOCKET listener_=INVALID_SOCKET;uint16_t port_=0;std::vector<Pending> pending_;
};
} // namespace nodus::contracts
