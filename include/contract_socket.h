#pragma once
// Connected-socket adapter, no listener or authentication policy inside the tool.
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
namespace nodus::contracts {
enum class WireKind:uint32_t { Offer=1,Accept=2,Result=3,Receipt=4 };
struct WireMessage {WireKind kind;Result result{};};
class ContractSocket {
public:
    explicit ContractSocket(SOCKET socket,size_t capacity=64):socket_(socket),capacity_(capacity){
        if(socket==INVALID_SOCKET || !capacity) throw std::invalid_argument("invalid contract socket");
#ifdef _WIN32
        u_long on=1;if(ioctlsocket(socket_,FIONBIO,&on)!=0) throw std::runtime_error("nonblocking socket refused");
#else
        int flags=fcntl(socket_,F_GETFL,0);
        if(flags<0 || fcntl(socket_,F_SETFL,flags|O_NONBLOCK)<0)throw std::runtime_error("nonblocking socket refused");
#endif
    }
    ~ContractSocket(){
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
    }
    ContractSocket(const ContractSocket&)=delete;
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
    using Frame=std::array<uint8_t,72>;
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
        auto kind=get(f,4,4);if(kind<1 || kind>4)return false;m.kind=WireKind(kind);
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
    SOCKET socket_;size_t capacity_;bool closed_=false;uint64_t last_offer_=0;
    std::deque<Frame> outgoing_;std::deque<WireMessage> incoming_;
    size_t sent_=0,read_count_=0;Frame read_{};
};
} // namespace nodus::contracts
