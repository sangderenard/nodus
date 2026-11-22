// nodus_mesh_keys: the mesh's key operators, as an operator runs them (KEY_CHANGE_PROTOCOL.md, workspace root). The
// work is nodus_mesh_operator (src/headless/nodus_mesh_crypto.cpp, the pinned OpenSSL); this prints what it did.
//
//   nodus_mesh_keys keygen <folder>                 a node's first key set (never replaces any)
//   nodus_mesh_keys root-keygen <folder>            the site root (keep the folder offline)
//   nodus_mesh_keys rotate <folder> [grace s]       the next epoch, resumable; writes rotation-<n>.rec
//   nodus_mesh_keys force <root folder> <out file> [<revoked kid hex>...] [--suites 1,2] [--replace <node.pub>]
//   nodus_mesh_keys accept <folder> <record file>   apply a rotation or force record
#include "nodus_headless_abi.h"

#include <cstdio>

int main(int argc, char** argv) {
    char report[1024] = {};
    const int ok = nodus_mesh_operator(argc - 1, argv + 1, report, sizeof(report));
    std::printf("%s\n", report);
    return ok ? 0 : 1;
}
