#include "common/tensors/abstraction/tensor_registry.h"
#include "common/tensors/abstraction/tensor_backend.h"

#include <cassert>
#include <iostream>

using namespace nodus::tensors;

struct DummyBackend final : public TensorBackend {
    const char* name() const override { return "dummy"; }
    AbstractTensorHandle create(const TensorDesc&) override { return {}; }
    void destroy(AbstractTensorHandle) override {}
    bool describe(AbstractTensorHandle, TensorDesc*) const override { return false; }
    bool get_item(AbstractTensorHandle,
                  const TensorIndexSpec&,
                  AbstractTensorHandle*,
                  TensorDesc*) override {
        return false;
    }
    bool set_item(AbstractTensorHandle,
                  const TensorIndexSpec&,
                  AbstractTensorHandle) override {
        return false;
    }
};

int main() {
    DummyBackend backend;
    register_backend(&backend, true);

    TensorBackend* found = find_backend("dummy");
    assert(found == &backend);
    assert(default_backend() == &backend);

    std::cout << "tensor_registry_test: ok\n";
    return 0;
}
