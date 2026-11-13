#include "tcp_byte_stream_tool.h"

#include "common/tensors/abstraction/tensor_abi.h"
#include "contract_listener.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <vector>

namespace nodus::net {
namespace {
std::uint64_t load_length(const std::array<std::uint8_t,8>& bytes){
    std::uint64_t value=0;
    for(std::size_t i=0;i<8;i++)value|=std::uint64_t(bytes[i])<<(8*i);
    return value;
}
void store_length(std::vector<std::uint8_t>& bytes,std::uint64_t value){
    for(std::size_t i=0;i<8;i++)bytes[i]=std::uint8_t(value>>(8*i));
}
}

struct TcpByteStreamTool::Impl {
    struct Outbound {std::vector<std::uint8_t> bytes;std::size_t sent=0;};
    struct Reader {
        SOCKET socket=INVALID_SOCKET;
        std::shared_ptr<const std::vector<std::uint8_t>> current;
        std::shared_ptr<const std::vector<std::uint8_t>> latest;
        std::size_t sent=0;
        std::uint64_t dropped=0;
    };
    explicit Impl(const TcpByteStreamConfig& config)
        :listener(config.port),maximum(config.maximum_frame_bytes),
         capacity(config.pending_frames),max_readers(config.broadcast_readers){}
    ~Impl(){
        if(peer!=INVALID_SOCKET)contracts::WireSocket::close_socket(peer);
        for(auto& reader:readers)contracts::WireSocket::close_socket(reader.socket);
    }
    contracts::ContractListener listener;
    SOCKET peer=INVALID_SOCKET;
    std::size_t maximum=0,capacity=0;
    std::size_t max_readers=0;
    std::vector<Reader> readers;
    std::deque<Outbound> outbound;
    std::deque<std::vector<std::uint8_t>> inbound;
    std::array<std::uint8_t,8> length_bytes{};
    std::size_t length_read=0,body_read=0;
    std::vector<std::uint8_t> body;
    bool uncertain=false;

    void close_peer(){
        if(peer!=INVALID_SOCKET)contracts::WireSocket::close_socket(peer);
        peer=INVALID_SOCKET;
        if(!outbound.empty()||length_read||body_read)uncertain=true;
        outbound.clear();body.clear();length_read=body_read=0;
    }
    void poll(){
        if(max_readers){
            listener.poll([this](SOCKET accepted,std::uint64_t,std::uint64_t){
                if(readers.size()>=max_readers)contracts::WireSocket::close_socket(accepted);
                else readers.push_back(Reader{accepted});
            });
            for(std::size_t i=0;i<readers.size();){
                auto& reader=readers[i];bool lost=false;
                for(int step=0;step<1;step++){
                    if(!reader.current){
                        reader.current=std::move(reader.latest);reader.sent=0;
                        if(!reader.current)break;
                    }
                    const auto left=reader.current->size()-reader.sent;
                    const auto n=::send(reader.socket,reinterpret_cast<const char*>(
                        reader.current->data()+reader.sent),
                        static_cast<int>(std::min<std::size_t>(left,65536)),0);
                    if(n<0){if(!contracts::WireSocket::would_block())lost=true;break;}
                    if(n==0){lost=true;break;}
                    reader.sent+=static_cast<std::size_t>(n);
                    if(reader.sent==reader.current->size()){
                        reader.current.reset();reader.sent=0;
                    }
                }
                if(lost){contracts::WireSocket::close_socket(reader.socket);
                    readers.erase(readers.begin()+i);}
                else ++i;
            }
            return;
        }
        listener.poll([this](SOCKET accepted,std::uint64_t,std::uint64_t){
            if(peer!=INVALID_SOCKET||uncertain)
                contracts::WireSocket::close_socket(accepted);
            else peer=accepted;
        });
        if(peer==INVALID_SOCKET||uncertain)return;
        // Bounded nonblocking work per table visit. A full inbound queue
        // stops recv so the kernel FIFO ultimately pressures the sender.
        for(int step=0;step<8&&!outbound.empty();step++){
            auto& frame=outbound.front();
            const auto left=frame.bytes.size()-frame.sent;
            const auto n=::send(peer,reinterpret_cast<const char*>(
                frame.bytes.data()+frame.sent),
                static_cast<int>(std::min<std::size_t>(left,65536)),0);
            if(n<0){if(!contracts::WireSocket::would_block())close_peer();break;}
            if(n==0){close_peer();break;}
            frame.sent+=static_cast<std::size_t>(n);
            if(frame.sent==frame.bytes.size())outbound.pop_front();
        }
        for(int step=0;step<8&&peer!=INVALID_SOCKET&&
                         inbound.size()<capacity;step++){
            if(length_read<8){
                const auto n=::recv(peer,reinterpret_cast<char*>(
                    length_bytes.data()+length_read),
                    static_cast<int>(8-length_read),0);
                if(n<0){if(!contracts::WireSocket::would_block())close_peer();break;}
                if(n==0){close_peer();break;}
                length_read+=static_cast<std::size_t>(n);
                if(length_read<8)continue;
                const auto size=load_length(length_bytes);
                if(size==0||size>maximum){close_peer();uncertain=true;break;}
                body.resize(static_cast<std::size_t>(size));
            }
            const auto n=::recv(peer,reinterpret_cast<char*>(body.data()+body_read),
                static_cast<int>(std::min<std::size_t>(body.size()-body_read,65536)),0);
            if(n<0){if(!contracts::WireSocket::would_block())close_peer();break;}
            if(n==0){close_peer();break;}
            body_read+=static_cast<std::size_t>(n);
            if(body_read==body.size()){
                inbound.push_back(std::move(body));
                body.clear();length_read=body_read=0;
            }
        }
    }
    void broadcast(std::shared_ptr<const std::vector<std::uint8_t>> frame){
        for(auto& reader:readers){
            if(!reader.current)reader.current=frame;
            else {if(reader.latest)reader.dropped++;reader.latest=frame;}
        }
    }
};

TcpByteStreamTool::TcpByteStreamTool(TcpByteStreamConfig config)
    :config_(std::move(config)){}
TcpByteStreamTool::~TcpByteStreamTool() noexcept{shutdown();}
void TcpByteStreamTool::initialize(const ToolInitContext&){
    if(impl_)return;
    try{impl_=std::make_unique<Impl>(config_);}
    catch(const std::exception& e){
        std::fprintf(stderr,"[tcp-byte-stream] %s: %s\n",config_.id.c_str(),e.what());
    }
}
void TcpByteStreamTool::shutdown(){impl_.reset();}
void TcpByteStreamTool::tick(double,HostAPI&){if(impl_)impl_->poll();}
std::uint16_t TcpByteStreamTool::listen_port() const noexcept{
    return impl_?impl_->listener.port():0;
}
bool TcpByteStreamTool::uncertain_delivery() const noexcept{
    return impl_&&impl_->uncertain;
}
void TcpByteStreamTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!impl_||!context.stack.raw)return;
    impl_->poll();
    auto& stack=*context.stack.raw;
    if(config_.broadcast_readers){
        nodus::tensors::AbstractTensorHandle input{};
        if(!raw_stack_pop_abstract_tensor(stack,input))return;
        NodusTensorDesc desc{};
        if(nodus_tensor_describe(input.id,&desc)!=NODUS_OK||
           desc.dtype!=NODUS_DTYPE_U8||desc.rank!=1||
           !desc.total_bytes||desc.total_bytes>impl_->maximum){
            raw_stack_push_abstract_tensor(stack,input);return;
        }
        auto frame=std::make_shared<std::vector<std::uint8_t>>(
            static_cast<std::size_t>(desc.total_bytes)+8);
        store_length(*frame,desc.total_bytes);
        if(nodus_tensor_read(input.id,0,frame->data()+8,desc.total_bytes)!=
           static_cast<std::int64_t>(desc.total_bytes)){
            raw_stack_push_abstract_tensor(stack,input);return;
        }
        impl_->broadcast(frame);
        impl_->poll();
        context.succeeded=true;
        return;
    }
    bool accepted_input=false;
    nodus::tensors::AbstractTensorHandle input{};
    if(raw_stack_pop_abstract_tensor(stack,input)){
        if(impl_->uncertain||impl_->peer==INVALID_SOCKET||
           impl_->outbound.size()>=impl_->capacity){
            raw_stack_push_abstract_tensor(stack,input);return;
        }
        NodusTensorDesc desc{};
        if(nodus_tensor_describe(input.id,&desc)!=NODUS_OK||
           desc.dtype!=NODUS_DTYPE_U8||desc.rank!=1||
           !desc.total_bytes||desc.total_bytes>impl_->maximum){
            raw_stack_push_abstract_tensor(stack,input);return;
        }
        Impl::Outbound frame;
        frame.bytes.resize(static_cast<std::size_t>(desc.total_bytes)+8);
        store_length(frame.bytes,desc.total_bytes);
        if(nodus_tensor_read(input.id,0,frame.bytes.data()+8,desc.total_bytes)!=
           static_cast<std::int64_t>(desc.total_bytes)){
            raw_stack_push_abstract_tensor(stack,input);return;
        }
        impl_->outbound.push_back(std::move(frame));
        impl_->poll();
        accepted_input=true;
    }
    if(impl_->inbound.empty()){
        context.succeeded=accepted_input;return;
    }
    auto& bytes=impl_->inbound.front();
    const std::uint64_t shape[1]={bytes.size()};
    const auto tensor=nodus_tensor_create(NODUS_DTYPE_U8,1,shape);
    if(!tensor){context.succeeded=accepted_input;return;}
    if(nodus_tensor_write(tensor,0,bytes.data(),bytes.size())!=
           static_cast<std::int64_t>(bytes.size())||
       !raw_stack_push_abstract_tensor(stack,{tensor})){
        nodus_tensor_destroy(tensor);
        context.succeeded=accepted_input;return;
    }
    impl_->inbound.pop_front();
    context.succeeded=true;
}

bool register_tcp_byte_stream(ToolRegistry& registry,TcpByteStreamConfig config){
    if(config.id.empty()||config.maximum_frame_bytes==0||
       config.maximum_frame_bytes>64*1024*1024||
       config.pending_frames==0||config.pending_frames>1024||
       config.broadcast_readers>64||
       (config.broadcast_readers&&config.pending_frames!=1))return false;
    ToolRegistry::Entry entry;
    entry.id=config.id;entry.name=config.broadcast_readers?"TCP broadcast exit":"TCP byte stream";
    entry.caps=ToolCaps::None;
    entry.factory=[config]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new TcpByteStreamTool(config),[](ITool* tool){delete tool;});
    };
    return registry.register_tool(std::move(entry));
}
} // namespace nodus::net
