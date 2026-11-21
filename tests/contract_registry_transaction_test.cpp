// CPU allocation-fault sweep of real in-image registry publication. No GPU.
#include "contract_negotiator_registry.h"
#include <cstdlib>
#include <iostream>
#include <new>

namespace {
bool injecting=false,saw_first_factory=false;
size_t remaining=0;
const std::string prefix="transaction.cpu.registry.long-name";
const std::string market_id=prefix+".market",worker_id=prefix+".worker";
void allocation_checkpoint() {
    if(!injecting)return;
    if(remaining--==0) {
        injecting=false;
        saw_first_factory|=tool_registry_global().find(market_id)!=nullptr &&
                           tool_registry_global().find(worker_id)==nullptr;
        throw std::bad_alloc();
    }
}
void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
}
void* operator new(size_t bytes){allocation_checkpoint();if(auto* p=std::malloc(bytes?bytes:1))return p;throw std::bad_alloc();}
void* operator new[](size_t bytes){return ::operator new(bytes);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,size_t)noexcept{std::free(p);}
void operator delete[](void* p,size_t)noexcept{std::free(p);}
int main() {
    using namespace nodus::contracts;
    try {
        (void)tool_registry_global();
        const auto offer=[](const Peer&,const std::vector<ContractNegotiator::Entry>&){return size_t(-1);};
        size_t failures=0;bool succeeded=false;
        for(size_t fail_at=0;fail_at<512;++fail_at) {
            remaining=fail_at;injecting=true;
            try {
                auto& owner=create_with_offer(prefix,8,offer,2);injecting=false;
                check(find(prefix)==&owner,"successful owner is canonical");
                check(tool_registry_global().find(market_id) && tool_registry_global().find(worker_id),"successful pair has both factories");
                // Teardown must not allocate after it starts removing factories.
                remaining=0;injecting=true;check(release(prefix),"pair release");injecting=false;
                succeeded=true;break;
            } catch(const std::bad_alloc&) {
                injecting=false;++failures;
                check(!find(prefix),"failed publication retained canonical owner");
                check(!tool_registry_global().find(market_id) && !tool_registry_global().find(worker_id),
                    "failed publication retained dangling or partial factory");
            }
        }
        check(succeeded && failures>3 && saw_first_factory,"sweep covered allocations before publication and after first factory");
        // A collision must never remove an unrelated existing factory.
        ToolRegistry::Entry existing;existing.id=worker_id;existing.name="pre-existing";
        existing.factory=[] {return std::unique_ptr<ITool,std::function<void(ITool*)>>(new NegotiatorTool(false),[](ITool* p){delete p;});};
        check(tool_registry_global().register_tool(std::move(existing)),"pre-existing factory");
        bool refused=false;try{create_with_offer(prefix,8,offer,2);}catch(const std::runtime_error&){refused=true;}
        check(refused && !find(prefix) && !tool_registry_global().find(market_id) &&
              tool_registry_global().find(worker_id),"collision preserves foreign factory");
        tool_registry_global().unregister_tool(worker_id);
        std::cout<<"PASS: "<<failures<<" creation allocation failures rolled back, including after first factory; release allocated nothing; foreign factory preserved\n";
        return 0;
    } catch(const std::exception& e){injecting=false;std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
