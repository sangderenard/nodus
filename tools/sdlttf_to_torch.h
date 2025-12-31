#pragma once

#include <string>
#include <stdexcept>
#if defined(__has_include)
# if __has_include(<torch/torch.h>)
#  include <torch/torch.h>
#  define NODUS_HAVE_LIBTORCH 1
# else
#  define NODUS_HAVE_LIBTORCH 0
namespace torch { class Tensor; }
# endif
#else
# include <torch/torch.h>
# define NODUS_HAVE_LIBTORCH 1
#endif

// Don't force SDL headers into every TU that includes this header.
// Instead, forward-declare the TTF_Font type and provide a small, portable
// color POD used by the API. The implementation file includes the SDL headers
// and converts between `SdlColor` and `SDL_Color`.

struct TTF_Font; // opaque from SDL_ttf

struct SdlColor {
    uint8_t r{255};
    uint8_t g{255};
    uint8_t b{255};
    uint8_t a{255};
};

// RAII guard for SDL / TTF initialization — declared here, defined in the .cpp
struct SdlTtfGuard {
    SdlTtfGuard();
    ~SdlTtfGuard();
    SdlTtfGuard(const SdlTtfGuard&) = delete;
    SdlTtfGuard& operator=(const SdlTtfGuard&) = delete;
};

// Lightweight font handle wrapper; defined in the .cpp to avoid SDL headers here.
struct FontHandle {
    TTF_Font* font{nullptr};
    explicit FontHandle(const char* path, int pt_size);
    ~FontHandle();
    FontHandle(const FontHandle&) = delete;
    FontHandle& operator=(const FontHandle&) = delete;
};

// Render a UTF-16 (std::u16string) string to a CPU torch::Tensor [H,W,4] uint8 RGBA.
// Uses SDL_ttf blended rendering; implementation converts SdlColor -> SDL_Color.
torch::Tensor render_text_utf16_rgba_u8(TTF_Font* font, const std::u16string& utf16, SdlColor fg, int wrap_width_px = 0);

// Float convenience: convert to float32 [0..1]
torch::Tensor render_text_utf16_rgba_f32(TTF_Font* font, const std::u16string& utf16, SdlColor fg, int wrap_width_px = 0);
