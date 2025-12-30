#include "value_types.h"
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>

// Minimal SHA256 (public-domain style small implementation)
// Adapted from small reference implementations for test purposes.
// Note: optimized security not required for test; just correctness.
typedef struct { uint32_t h[8]; uint8_t buf[64]; uint64_t len; size_t buf_len; } sha256_ctx;
static const uint32_t K256[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static uint32_t rotr(uint32_t x,int r){return (x>>r)|(x<<(32-r));}
static void sha256_init(sha256_ctx* c){
  c->h[0]=0x6a09e667; c->h[1]=0xbb67ae85; c->h[2]=0x3c6ef372; c->h[3]=0xa54ff53a;
  c->h[4]=0x510e527f; c->h[5]=0x9b05688c; c->h[6]=0x1f83d9ab; c->h[7]=0x5be0cd19;
  c->len=0; c->buf_len=0; memset(c->buf,0,64);
}

static void sha256_compress(sha256_ctx* c,const uint8_t* p){
  uint32_t w[64];
  for(int i=0;i<16;i++){
    w[i] = (uint32_t)p[i*4]<<24 | (uint32_t)p[i*4+1]<<16 | (uint32_t)p[i*4+2]<<8 | (uint32_t)p[i*4+3];
  }
  for(int i=16;i<64;i++){
    uint32_t s0 = rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
    uint32_t s1 = rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
    w[i] = w[i-16] + s0 + w[i-7] + s1;
  }
  uint32_t a=c->h[0],b=c->h[1],d=c->h[3],e=c->h[4],f=c->h[5],g=c->h[6],h=c->h[7],c2=c->h[2];
  uint32_t A=a,B=b,C=c2,D=d,E=e,F=f,G=g,H=h;
  for(int i=0;i<64;i++){
    uint32_t S1 = rotr(E,6)^rotr(E,11)^rotr(E,25);
    uint32_t ch = (E & F) ^ ((~E) & G);
    uint32_t temp1 = H + S1 + ch + K256[i] + w[i];
    uint32_t S0 = rotr(A,2)^rotr(A,13)^rotr(A,22);
    uint32_t maj = (A & B) ^ (A & C) ^ (B & C);
    uint32_t temp2 = S0 + maj;
    H = G; G = F; F = E; E = D + temp1; D = C; C = B; B = A; A = temp1 + temp2;
  }
  c->h[0]+=A; c->h[1]+=B; c->h[2]+=C; c->h[3]+=D; c->h[4]+=E; c->h[5]+=F; c->h[6]+=G; c->h[7]+=H;
}

static void sha256_update(sha256_ctx* c,const uint8_t* data,size_t len){
  c->len += (uint64_t)len * 8;
  while(len){
    size_t tocopy = std::min(len, 64 - c->buf_len);
    memcpy(c->buf + c->buf_len, data, tocopy);
    c->buf_len += tocopy; data += tocopy; len -= tocopy;
    if(c->buf_len == 64){ sha256_compress(c, c->buf); c->buf_len = 0; }
  }
}

static void sha256_final(sha256_ctx* c,uint8_t out[32]){
  size_t i = c->buf_len;
  c->buf[i++] = 0x80;
  if(i > 56){ while(i<64) c->buf[i++]=0; sha256_compress(c,c->buf); i=0; }
  while(i<56) c->buf[i++]=0;
  // append length in bits
  for(int j=7;j>=0;j--) c->buf[i++] = (uint8_t)((c->len >> (j*8)) & 0xFF);
  sha256_compress(c,c->buf);
  for(int k=0;k<8;k++){ out[k*4+0] = (uint8_t)((c->h[k]>>24)&0xFF); out[k*4+1] = (uint8_t)((c->h[k]>>16)&0xFF); out[k*4+2] = (uint8_t)((c->h[k]>>8)&0xFF); out[k*4+3] = (uint8_t)(c->h[k]&0xFF); }
}

static std::string sha256_hex(const uint8_t digest[32]){
  static const char* hex = "0123456789abcdef";
  std::string s; s.reserve(64);
  for(int i=0;i<32;i++){ s.push_back(hex[digest[i]>>4]); s.push_back(hex[digest[i]&0xF]); }
  return s;
}

int main(){
  const size_t N = 1024;
  // Create source backend and fill with pattern
  gp_mem_backend_handle_t h_src = gp_mem_backend_create_host(N);
  if(!h_src){ std::cerr<<"failed create src backend\n"; return 2; }
  void* m = gp_mem_backend_map(h_src);
  if(!m){ std::cerr<<"failed map src\n"; return 3; }
  for(size_t i=0;i<N;i++) ((uint8_t*)m)[i] = static_cast<uint8_t>((i*31 + 7) & 0xFF);
  gp_mem_backend_unmap(h_src);

  // Compute original SHA
  sha256_ctx ctx; uint8_t digest[32]; sha256_init(&ctx);
  void* rm = gp_mem_backend_map(h_src);
  sha256_update(&ctx, reinterpret_cast<const uint8_t*>(rm), N);
  gp_mem_backend_unmap(h_src);
  sha256_final(&ctx,digest);
  std::string orig_sha = sha256_hex(digest);
  std::cerr << "orig sha=" << orig_sha << "\n";

  // Initialize a RawStackFrame and transfer src->frame backend via gp_mem_backend_transfer
  RawStackFrame* frame = new RawStackFrame();
  if(!raw_stack_init_frame(*frame, N)) { std::cerr<<"init frame failed\n"; return 4; }
  // perform transfer
  if(!gp_mem_backend_transfer(h_src, frame->backend, 0, 0, N)){ std::cerr<<"transfer failed\n"; return 5; }
  // set type mask to uint8
  ValueTypeId vt_u8 = ValueTypeRegistry::global().builtin(VT_UINT8);
  for(size_t i=0;i<N;i++) frame->types_per_byte[i] = vt_u8;
  frame->byte_count = N;

  // Serialize mid-transfer
  size_t need = 0; if(!gp_raw_stack_frame_serialized_size(frame, &need)){ std::cerr<<"serialized_size fail\n"; return 6; }
  std::vector<uint8_t> ser(need);
  size_t written = 0; if(!gp_raw_stack_frame_serialize(frame, ser.data(), ser.size(), &written)){ std::cerr<<"serialize fail\n"; return 7; }
  std::cerr<<"serialized mid written="<<written<<"\n";

  // Deserialize into a new frame on another backend
  RawStackFrame* frame2 = new RawStackFrame();
  if(!gp_raw_stack_frame_deserialize(frame2, ser.data(), written)){ std::cerr<<"deserialize fail\n"; return 8; }

  // Now transfer frame2->dst backend
  gp_mem_backend_handle_t h_dst = gp_mem_backend_create_host(N);
  if(!h_dst){ std::cerr<<"create dst failed\n"; return 9; }
  // Pop all bytes from frame2 into the destination backend preserving original order.
  while (frame2->byte_count > 0) {
    size_t dst_off = frame2->byte_count - 1; // write the top byte into its original index
    if (!gp_raw_stack_frame_pop_into_backend(frame2, h_dst, dst_off, vt_u8)) { std::cerr<<"pop_into_backend failed\n"; return 10; }
  }

  // Compute SHA of dst
  sha256_ctx ctx2; uint8_t digest2[32]; sha256_init(&ctx2);
  void* rm2 = gp_mem_backend_map(h_dst);
  sha256_update(&ctx2, reinterpret_cast<const uint8_t*>(rm2), N);
  gp_mem_backend_unmap(h_dst);
  sha256_final(&ctx2,digest2);
  std::string final_sha = sha256_hex(digest2);
  std::cerr << "final sha=" << final_sha << "\n";

  // Verify
  if(final_sha != orig_sha){ std::cerr<<"sha mismatch\n"; return 11; }

  // cleanup
  raw_stack_destroy_frame(frame);
  raw_stack_destroy_frame(frame2);
  gp_mem_backend_release(h_src);
  gp_mem_backend_release(h_dst);

  std::cout<<"edge_backend_coordination_test: PASS\n";
  return 0;
}
