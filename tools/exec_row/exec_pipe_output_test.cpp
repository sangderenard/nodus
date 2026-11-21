#include "exec_pipe.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#define NODUS_BINARY_STDOUT() _setmode(_fileno(stdout), _O_BINARY)
#else
#define NODUS_BINARY_STDOUT() ((void)0)
#endif
#include <fcntl.h>
using namespace nodus::exec;
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv) {
 if(argc>1 && std::string(argv[1])=="emit") { NODUS_BINARY_STDOUT(); for(int i=0;i<1048576;i++) if(std::putchar(i%251)==EOF) return 2; return 0; }
 auto check=[](bool yes,const char*name){ if(!yes){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);} std::printf("PASS %s\n",name); };
 Program p{std::filesystem::absolute(argv[0]).string(),{"emit"},std::filesystem::current_path().string(),{}, {}};
 Child c; std::string why;
 check(c.start(p,why,false,4096),"start bounded stdout"); c.finish_input();
 std::this_thread::sleep_for(std::chrono::milliseconds(100));
 check(c.buffered_output()<=4096 && c.running(),"bounded backpressure");
 size_t total=0; auto deadline=Clock::now()+std::chrono::seconds(10);
 while(!c.output_drained()) {
  if(Clock::now()>deadline) return 4;
  std::string part; c.take(part,333);
  for(unsigned char b:part) { if(b!=total%251) return 3; ++total; }
  if(part.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
 }
 check(total==1048576,"exact bytes through partial reads and EOF");
 while(c.running()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
 check(c.exit_code()==0,"successful child exit");
 check(c.start(p,why,false,17),"restart tiny bound"); c.finish_input();
 std::this_thread::sleep_for(std::chrono::milliseconds(100));
 auto start=Clock::now(); c.stop();
 check(Clock::now()-start<std::chrono::seconds(2),"cancel while backpressured");
 check(c.start(p,why),"legacy unbounded mode");c.finish_input();
 while(!c.finished()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
 std::string all;c.take(all);check(all.size()==1048576,"legacy span output preserved");
 if(argc==3) {
  Program decode{argv[1],{"-nostdin","-v","error","-i",argv[2],"-f","f32le","-ar","48000","-ac","2","pipe:1"},std::filesystem::current_path().string(),{}, {}};
  check(c.start(decode,why,false,65536),"real FFmpeg bounded decoder");c.finish_input();
  size_t bytes=0;deadline=Clock::now()+std::chrono::seconds(30);
  while(!c.output_drained()) {
   if(Clock::now()>deadline) return 5;
   if(c.buffered_output()>65536) return 6;
   std::string part;c.take(part,2048);bytes+=part.size();
   if(part.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  while(c.running()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  check(c.exit_code()==0 && bytes==180ull*48000*2*4,"180s decode exceeds old 64MiB without full buffering");
 }

}
