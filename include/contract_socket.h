#pragma once
// Connected-socket adapters (ContractSocket = host side, EngineSocket = engine side),
// no listener or authentication policy inside this file (see contract_listener.h).
// Caller authenticates/authorizes the connection before providing its socket.
// Fixed 72-byte little-endian v1 frames; bounded queues; never blocks a graph tick.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#else
#include <sys/socket.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
using SOCKET=int;
static constexpr SOCKET INVALID_SOCKET=-1;
#endif
#include "contract_negotiator.h"
#include <array>
#include <deque>
#include <cstring>
#include <stdexcept>
namespace nodus::contracts {
enum class WireKind:uint32_t { Offer=1,Accept=2,Result=3,Receipt=4,
    Hello=5,    // sender->nodus: work.id=role(1 host,2 engine), work.engine=participant id, work.revision=2 (version)
    Submit=6,   // engine->nodus: work=Contract
    Admitted=7, // nodus->engine: work.id=contract id
    Route=8,    // nodus->engine: full Result
    Decision=9  // engine->nodus: work.id, loan.generation, payload=1 commit,0 hold,2 reject
};
struct WireMessage {WireKind kind;Result result{};};
// Shared nonblocking, bounded-queue frame transport. Not used polymorphically.
class WireSocket {
public:
    using Frame=std::array<uint8_t,72>;
    bool closed() const{return closed_;}
    bool send_message(WireMessage message){
        if(closed_ || outgoing_.size()>=capacity_)return false;
        outgoing_.push_back(encode(message));return true;
    }
    bool receive(WireMessage& message){
        if(incoming_.empty())return false;
        message=incoming_.front();incoming_.pop_front();return true;
    }
    void poll(){
        if(closed_)return;
        // Fixed work bound even if a peer continuously sends data.
        for(size_t i=0;i<capacity_ && !outgoing_.empty();++i){
            auto& frame=outgoing_.front();
            int flags=0;
#ifdef MSG_NOSIGNAL
            flags=MSG_NOSIGNAL;
#endif
            int n=::send(socket_,reinterpret_cast<const char*>(frame.data()+sent_),int(frame.size()-sent_),flags);
            if(n<0){if(!would_block())closed_=true;break;}if(n==0){closed_=true;break;}
            sent_+=size_t(n);if(sent_==frame.size()){outgoing_.pop_front();sent_=0;}
        }
        for(size_t i=0;i<capacity_ && incoming_.size()<capacity_;++i){
            int n=::recv(socket_,reinterpret_cast<char*>(read_.data()+read_count_),int(read_.size()-read_count_),0);
            if(n<0){if(!would_block())closed_=true;break;}if(n==0){closed_=true;break;}
            read_count_+=size_t(n);
            if(read_count_==read_.size()){
                WireMessage m{};if(!decode(read_,m)){closed_=true;break;}
                incoming_.push_back(m);read_count_=0;
            }
        }
    }
    static void put(Frame& f,size_t offset,uint64_t value,size_t bytes){for(size_t i=0;i<bytes;++i)f[offset+i]=uint8_t(value>>(8*i));}
    static uint64_t get(const Frame& f,size_t offset,size_t bytes){uint64_t v=0;for(size_t i=0;i<bytes;++i)v|=uint64_t(f[offset+i])<<(8*i);return v;}
    static Frame encode(const WireMessage& m){
        Frame f{};put(f,0,0x31434E4E,4);put(f,4,uint32_t(m.kind),4);
        const auto& l=m.result.loan;
        uint64_t values[]={l.work.id,l.work.engine,l.work.revision,l.work.payload,l.generation,l.host,l.deadline,m.result.payload};
        for(size_t i=0;i<8;++i)put(f,8+8*i,values[i],8);return f;
    }
    static bool decode(const Frame& f,WireMessage& m){
        if(get(f,0,4)!=0x31434E4E)return false;
        auto kind=get(f,4,4);if(kind<1 || kind>9)return false;m.kind=WireKind(kind);
        uint64_t v[8];for(size_t i=0;i<8;++i)v[i]=get(f,8+8*i,8);
        m.result={{{v[0],v[1],v[2],v[3]},v[4],v[5],v[6]},v[7]};return true;
    }
    static bool would_block(){
#ifdef _WIN32
        return WSAGetLastError()==WSAEWOULDBLOCK;
#else
        return errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR;
#endif
    }
    static bool set_nonblocking(SOCKET s){
#ifdef _WIN32
        u_long on=1;return ioctlsocket(s,FIONBIO,&on)==0;
#else
        int flags=fcntl(s,F_GETFL,0);
        return flags>=0 && fcntl(s,F_SETFL,flags|O_NONBLOCK)>=0;
#endif
    }
    static void close_socket(SOCKET s){
#ifdef _WIN32
        closesocket(s);
#else
        close(s);
#endif
    }
protected:
    WireSocket(SOCKET socket,size_t capacity):socket_(socket),capacity_(capacity){
        if(socket==INVALID_SOCKET || !capacity) throw std::invalid_argument("invalid contract socket");
        if(!set_nonblocking(socket_)) throw std::runtime_error("nonblocking socket refused");
    }
    ~WireSocket(){close_socket(socket_);}
    WireSocket(const WireSocket&)=delete;
    WireSocket& operator=(const WireSocket&)=delete;
    void fail(){closed_=true;}
    SOCKET socket_;size_t capacity_;bool closed_=false;
    std::deque<Frame> outgoing_;std::deque<WireMessage> incoming_;
    size_t sent_=0,read_count_=0;Frame read_{};
};
// HOST side: a participant that accepts offers and returns results.
class ContractSocket : public WireSocket {
public:
    explicit ContractSocket(SOCKET socket,size_t capacity=64):WireSocket(socket,capacity){}
    Exchange exchange(const Loan* offer){
        if(offer && offer->generation!=last_offer_){
            if(send_message({WireKind::Offer,{*offer,0}}))last_offer_=offer->generation;
        }
        poll();Exchange reply{};WireMessage m{};
        // One response token per visit; acceptance and result may arrive on
        // successive visits. Each has its own generation identity.
        if(receive(m)){
            if(m.kind==WireKind::Accept)reply.accepted_generation=m.result.loan.generation;
            else if(m.kind==WireKind::Result){reply.has_result=true;reply.result=m.result;}
        }
        return reply;
    }
    bool receipt(uint64_t generation,bool accepted){
        WireMessage m{WireKind::Receipt,{}};m.result.loan.generation=generation;m.result.payload=accepted?1:0;
        return send_message(m);
    }
private:
    uint64_t last_offer_=0;
};
// ENGINE side: a Producer backed by a remote engine. Wire it as
//   Producer{id,[&](const Result& r){return s.route(r)==1;},[&](Contract& c){return s.request(c);},[&](uint64_t i){s.admitted(i);}}
// route() returns -1 for a rejected result; the owner then calls
// ContractNegotiator::reject_pending(id,generation) after the turn, as MailHost does.
// A peer exceeding `capacity` unadmitted Submits is a protocol violation: the socket closes.
class EngineSocket : public WireSocket {
public:
    explicit EngineSocket(SOCKET socket,size_t capacity=64):WireSocket(socket,capacity){}
    // Peek: oldest Submit not yet admitted; stays queued until admitted().
    bool request(Contract& out){
        pump();if(submits_.empty())return false;
        out=submits_.front();return true;
    }
    void admitted(uint64_t id){
        for(auto it=submits_.begin();it!=submits_.end();++it)if(it->id==id){submits_.erase(it);break;}
        WireMessage m{WireKind::Admitted,{}};m.result.loan.work.id=id;
        send_message(m);poll();
    }
    // Sends Route once per (contract id, generation); 1 commit, 0 hold/none yet, -1 reject.
    int route(const Result& r){
        pump();
        const uint64_t id=r.loan.work.id,gen=r.loan.generation;
        if(!routed_ || routed_id_!=id || routed_gen_!=gen){
            if(!send_message({WireKind::Route,r}))return 0;
            routed_=true;routed_id_=id;routed_gen_=gen;poll();
        }
        for(auto it=decisions_.begin();it!=decisions_.end();++it)if(it->id==id && it->generation==gen){
            uint64_t d=it->value;decisions_.erase(it);
            if(d==1 || d==2)routed_=false;
            return d==1?1:(d==2?-1:0);
        }
        return 0;
    }
    void poll(){WireSocket::poll();}
private:
    struct Decided{uint64_t id,generation,value;};
    void pump(){
        WireSocket::poll();WireMessage m{};
        while(receive(m)){
            if(m.kind==WireKind::Submit){
                const Contract& c=m.result.loan.work;bool dup=!c.id;
                for(auto& s:submits_)dup|=s.id==c.id;
                if(dup)continue;
                if(submits_.size()>=capacity_){fail();return;}
                submits_.push_back(c);
            }else if(m.kind==WireKind::Decision){
                if(decisions_.size()>=capacity_)decisions_.pop_front();
                decisions_.push_back({m.result.loan.work.id,m.result.loan.generation,m.result.payload});
            }
        }
    }
    std::deque<Contract> submits_;std::deque<Decided> decisions_;
    bool routed_=false;uint64_t routed_id_=0,routed_gen_=0;
};
} // namespace nodus::contracts
