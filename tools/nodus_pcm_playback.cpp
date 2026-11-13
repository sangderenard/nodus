#include "nodus_headless_abi.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--device")) {
        std::fprintf(stderr, "usage: nodus_pcm_playback --device <SDL playback ID or 0=default> < stereo-s16le-48000.pcm\n");
        return 2;
    }
    char* end = nullptr; errno = 0;
    const auto device = std::strtoul(argv[2], &end, 10);
    if (errno || !*argv[2] || *end || argv[2][0] == '-' || device > UINT32_MAX) return 2;
    return nodus_headless_pcm_playback_stdin(static_cast<uint32_t>(device));
}
