#include "tcp_byte_stream_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "contract_socket.h"
#include "tool_registry.h"

#include <array>
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
bool send_all(SOCKET socket,const void* data,std::size_t bytes){
    auto* p=static_cast<const char*>(data);
    while(bytes){
        const int n=::send(socket,p,static_cast<int>(bytes),0);
        if(n<=0)return false;
        p+=n;bytes-=static_cast<std::size_t>(n);
    }
    return true;
}
bool receive_all(SOCKET socket,void* data,std::size_t bytes){
    auto* p=static_cast<char*>(data);
    while(bytes){
        const int n=::recv(socket,p,static_cast<int>(bytes),0);
        if(n<=0)return false;
        p+=n;bytes-=static_cast<std::size_t>(n);
    }
    return true;
}
void free_frame(RawStackFrame& frame){
    raw_stack_free_mask(frame);
    if(frame.backend)gp_mem_backend_release(frame.backend);
}
}

int main(){
    nodus::net::TcpByteStreamConfig config;
    config.id="nodus.tcp.byte_stream.proof";
    config.maximum_frame_bytes=1024;
    config.pending_frames=2;
    assert(nodus::net::register_tcp_byte_stream(tool_registry_global(),config));
    auto tool=tool_registry_global().create(config.id);
    auto* stream=dynamic_cast<nodus::net::TcpByteStreamTool*>(tool.get());
    assert(stream);
    tool->initialize({});
    assert(stream->listen_port()!=0);
    assert(tool->port_spec(0).kind==ToolPortKind::Argument);
    assert(tool->port_spec(1).kind==ToolPortKind::Return);

    RawStackFrame frame{};
    assert(raw_stack_init_frame(frame,4096));
    const std::array<std::uint8_t,5> outbound{'w','o','r','l','d'};
    const std::uint64_t shape[1]={outbound.size()};
    const auto tensor=nodus_tensor_create(NODUS_DTYPE_U8,1,shape);
    assert(tensor&&nodus_tensor_write(tensor,0,outbound.data(),outbound.size())==5);
    assert(raw_stack_push_abstract_tensor(frame,{tensor}));
    ToolStackContext context{};context.stack.raw=&frame;
    tool->execute_stack(context);
    assert(!context.succeeded); // input remains until a peer is admitted

    SOCKET peer=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    assert(peer!=INVALID_SOCKET);
    const int receive_timeout_ms=2000;
    assert(setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,
        reinterpret_cast<const char*>(&receive_timeout_ms),
        sizeof(receive_timeout_ms))==0);
    sockaddr_in address{};address.sin_family=AF_INET;
    address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    address.sin_port=htons(stream->listen_port());
    assert(connect(peer,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0);
    nodus::contracts::WireMessage hello{};
    hello.kind=nodus::contracts::WireKind::Hello;
    hello.result.loan.work.id=1;
    hello.result.loan.work.engine=42;
    hello.result.loan.work.revision=2;
    const auto hello_frame=nodus::contracts::WireSocket::encode(hello);
    assert(send_all(peer,hello_frame.data(),hello_frame.size()));
    for(int i=0;i<100&&!context.succeeded;i++)tool->execute_stack(context);
    assert(context.succeeded);
    std::array<std::uint8_t,13> received{};
    assert(receive_all(peer,received.data(),received.size()));
    assert(received[0]==5&&received[1]==0);
    assert(std::memcmp(received.data()+8,outbound.data(),5)==0);
    nodus_tensor_destroy(tensor);

    const std::array<std::uint8_t,4> inbound{'p','a','t','h'};
    std::array<std::uint8_t,12> wire{};wire[0]=4;
    std::memcpy(wire.data()+8,inbound.data(),inbound.size());
    assert(send_all(peer,wire.data(),wire.size()));
    context.succeeded=false;
    for(int i=0;i<100&&!context.succeeded;i++)tool->execute_stack(context);
    assert(context.succeeded);
    nodus::tensors::AbstractTensorHandle output{};
    assert(raw_stack_pop_abstract_tensor(frame,output));
    std::array<std::uint8_t,4> recovered{};
    assert(nodus_tensor_read(output.id,0,recovered.data(),recovered.size())==4);
    assert(recovered==inbound);
    nodus_tensor_destroy(output.id);
    nodus::contracts::WireSocket::close_socket(peer);
    tool->shutdown();
    free_frame(frame);
    nodus::net::TcpByteStreamConfig broadcast_config;
    broadcast_config.id="nodus.tcp.broadcast.proof";
    broadcast_config.maximum_frame_bytes=1024;
    broadcast_config.pending_frames=1;
    broadcast_config.broadcast_readers=2;
    assert(nodus::net::register_tcp_byte_stream(tool_registry_global(),broadcast_config));
    auto broadcast=tool_registry_global().create(broadcast_config.id);
    auto* exit=dynamic_cast<nodus::net::TcpByteStreamTool*>(broadcast.get());
    assert(exit);
    broadcast->initialize({});
    assert(broadcast->port_count()==1);
    assert(broadcast->port_spec(0).kind==ToolPortKind::Argument);
    assert(broadcast->port_spec(1).count==0);
    RawStackFrame broadcast_stack{};
    assert(raw_stack_init_frame(broadcast_stack,4096));
    ToolStackContext broadcast_context{};broadcast_context.stack.raw=&broadcast_stack;
    std::array<SOCKET,2> readers{};
    for(std::size_t i=0;i<readers.size();i++){
        readers[i]=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
        assert(readers[i]!=INVALID_SOCKET);
        assert(setsockopt(readers[i],SOL_SOCKET,SO_RCVTIMEO,
            reinterpret_cast<const char*>(&receive_timeout_ms),sizeof(receive_timeout_ms))==0);
        address.sin_port=htons(exit->listen_port());
        assert(connect(readers[i],reinterpret_cast<sockaddr*>(&address),sizeof(address))==0);
        auto greeting=hello;greeting.result.loan.work.engine=100+i;
        const auto encoded=nodus::contracts::WireSocket::encode(greeting);
        assert(send_all(readers[i],encoded.data(),encoded.size()));
    }
    for(int i=0;i<100;i++)broadcast->execute_stack(broadcast_context);
    const std::array<std::uint8_t,6> picture{'f','r','a','m','e','1'};
    const std::uint64_t picture_shape[1]={picture.size()};
    const auto picture_tensor=nodus_tensor_create(NODUS_DTYPE_U8,1,picture_shape);
    assert(picture_tensor&&nodus_tensor_write(picture_tensor,0,picture.data(),picture.size())==6);
    assert(raw_stack_push_abstract_tensor(broadcast_stack,{picture_tensor}));
    broadcast->execute_stack(broadcast_context);
    assert(broadcast_context.succeeded);
    for(auto reader:readers){
        std::array<std::uint8_t,14> bytes{};
        assert(receive_all(reader,bytes.data(),bytes.size()));
        assert(bytes[0]==picture.size());
        assert(std::memcmp(bytes.data()+8,picture.data(),picture.size())==0);
    }
    nodus_tensor_destroy(picture_tensor);
    for(auto reader:readers)nodus::contracts::WireSocket::close_socket(reader);
    broadcast->shutdown();
    free_frame(broadcast_stack);
    std::cout<<"tcp_byte_stream_roundtrip=1\n";
    return 0;
}
