#include "mem_backend.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <filesystem>

int main() {
    const char* dir = "./.test_manifest";
    std::filesystem::create_directories(dir);

    gp_mem_backend_handle_t fsb = gp_mem_backend_create_filesystem(dir, 0);
    if (!fsb) {
        std::cerr << "failed to create fs backend\n";
        return 2;
    }
    const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(fsb);
    if (!vt || !vt->alloc || !vt->copy_to_backend) {
        std::cerr << "fs backend missing vtable ops\n";
        return 3;
    }

    // allocate two file-backed buffers and write some data
    gp_mem_backend_handle_t h1 = vt->alloc(fsb, 1024, 1);
    gp_mem_backend_handle_t h2 = vt->alloc(fsb, 2048, 1);
    if (!h1 || !h2) { std::cerr << "alloc failed\n"; return 4; }

    std::vector<uint8_t> d1(1024, 0xAA);
    std::vector<uint8_t> d2(2048, 0x55);
    if (!vt->copy_to_backend(h1, 0, d1.data(), d1.size())) { std::cerr << "copy1 failed\n"; return 5; }
    if (!vt->copy_to_backend(h2, 0, d2.data(), d2.size())) { std::cerr << "copy2 failed\n"; return 6; }

    gp_mem_backend_handle_t arr[2] = { h1, h2 };
    std::string manifest_path = std::string(dir) + "/manifest.txt";
    if (!gp_mem_backend_write_manifest(manifest_path.c_str(), arr, 2)) {
        std::cerr << "write manifest failed\n"; return 7;
    }

    gp_mem_backend_handle_t* out_bufs = nullptr;
    size_t out_count = 0;
    if (!gp_mem_backend_read_manifest_and_allocate(fsb, manifest_path.c_str(), &out_bufs, &out_count)) {
        std::cerr << "read manifest failed\n"; return 8;
    }
    if (out_count != 2) { std::cerr << "unexpected out_count=" << out_count << "\n"; return 9; }

    std::cerr << "read manifest OK, out_bufs=" << (void*)out_bufs << " count=" << out_count << "\n";

    // verify sizes by inspecting native file paths
    for (size_t i = 0; i < out_count; ++i) {
        uintptr_t nh = gp_mem_backend_native_handle(out_bufs[i]);
        if (!nh) { std::cerr << "native handle missing\n"; return 10; }
        const char* path = reinterpret_cast<const char*>(nh);
        std::error_code ec;
        auto sz = std::filesystem::file_size(path, ec);
        if (ec) { std::cerr << "stat failed for " << path << "\n"; return 11; }
        if (i == 0 && sz != 1024) { std::cerr << "size mismatch0="<<sz<<"\n"; return 12; }
        if (i == 1 && sz != 2048) { std::cerr << "size mismatch1="<<sz<<"\n"; return 13; }
    }

    // cleanup
    std::cerr << "cleanup: releasing imported buffers\n";
    for (size_t i = 0; i < out_count; ++i) gp_mem_backend_release(out_bufs[i]);
    std::cerr << "cleanup: released imported buffers\n";
    std::free(out_bufs);
    std::cerr << "cleanup: freed out_bufs array\n";
    gp_mem_backend_release(h1);
    std::cerr << "cleanup: released h1\n";
    gp_mem_backend_release(h2);
    std::cerr << "cleanup: released h2\n";
    gp_mem_backend_release(fsb);
    std::cerr << "cleanup: released fsb\n";

    return 0;
}
