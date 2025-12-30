#include "sdlttf_to_torch.h"
#include <cstring>
#include <locale>
#include <codecvt>

// Implementation includes SDL3 headers when available; otherwise provide
// compile-time stubs so the project can build even when SDL_ttf isn't
// present on the system. This file now targets SDL3 only.
#if defined(__has_include)
#  if __has_include(<SDL3/SDL.h>) && __has_include(<SDL3/SDL_ttf.h>)
#    include <SDL3/SDL.h>
#    include <SDL3/SDL_ttf.h>
#    define NODUS_HAVE_SDLTTF 1
#  endif
#endif

#if defined(NODUS_HAVE_SDLTTF)

SdlTtfGuard::SdlTtfGuard() {
    if (SDL_WasInit(0) == 0) {
        if (SDL_Init(0) != 0) throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }
    if (TTF_WasInit() == 0) {
        if (TTF_Init() != 0) throw std::runtime_error(std::string("TTF_Init failed: ") + TTF_GetError());
    }
}

SdlTtfGuard::~SdlTtfGuard() {
    if (TTF_WasInit()) TTF_Quit();
}

FontHandle::FontHandle(const char* path, int pt_size) {
    font = TTF_OpenFont(path, pt_size);
    if (!font) throw std::runtime_error(std::string("TTF_OpenFont failed: ") + TTF_GetError());
}

FontHandle::~FontHandle() {
    if (font) TTF_CloseFont(font);
}

// Helper to convert SdlColor -> SDL_Color
static SDL_Color toSDLColor(const SdlColor& c) {
    SDL_Color s; s.r = c.r; s.g = c.g; s.b = c.b; s.a = c.a; return s;
}

#else // no SDL_ttf available: provide stubs that throw

SdlTtfGuard::SdlTtfGuard() {
    // No-op guard; actual usage will fail when attempting to open fonts.
}

SdlTtfGuard::~SdlTtfGuard() {}

FontHandle::FontHandle(const char* /*path*/, int /*pt_size*/) {
    throw std::runtime_error("SDL_ttf not available at compile time");
}

FontHandle::~FontHandle() {}

static void UNUSED_toSDLColor_fallback() {}

#endif

#if defined(NODUS_HAVE_SDLTTF)

// Helper: render using SDL_ttf unicode API (Uint16*). If wrap is used, use wrapped variant.
torch::Tensor render_text_utf16_rgba_u8(TTF_Font* font, const std::u16string& utf16, SdlColor fg, int wrap_width_px) {
    if (!font) throw std::invalid_argument("font is null");
    if (utf16.empty()) {
        return torch::empty({0, 0, 4}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    }

    // SDL_ttf provides TTF_RenderUNICODE_Blended which expects Uint16* (platform ordering)
    SDL_Surface* raw = nullptr;
    // Cast char16_t to Uint16 where sizes match
    static_assert(sizeof(char16_t) == sizeof(Uint16), "char16_t must be 16 bits");
    const Uint16* u16ptr = reinterpret_cast<const Uint16*>(utf16.data());
    SDL_Color sfg = toSDLColor(fg);
    if (wrap_width_px > 0) {
        raw = TTF_RenderUNICODE_Blended_Wrapped(font, u16ptr, sfg, static_cast<Uint32>(wrap_width_px));
    } else {
        raw = TTF_RenderUNICODE_Blended(font, u16ptr, sfg);
    }
    if (!raw) throw std::runtime_error(std::string("TTF_RenderUNICODE_Blended failed: ") + TTF_GetError());

    SDL_Surface* surf = SDL_ConvertSurfaceFormat(raw, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(raw);
    if (!surf) throw std::runtime_error(std::string("SDL_ConvertSurfaceFormat failed: ") + SDL_GetError());

    const int w = surf->w;
    const int h = surf->h;
    const int pitch = surf->pitch;
    const int row_bytes = w * 4;

    if (SDL_MUSTLOCK(surf)) {
        if (SDL_LockSurface(surf) != 0) {
            SDL_FreeSurface(surf);
            throw std::runtime_error(std::string("SDL_LockSurface failed: ") + SDL_GetError());
        }
    }

    auto out = torch::empty({h, w, 4}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    uint8_t* dst = out.data_ptr<uint8_t>();
    const uint8_t* src = static_cast<const uint8_t*>(surf->pixels);

    for (int y = 0; y < h; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * row_bytes, src + static_cast<size_t>(y) * pitch, static_cast<size_t>(row_bytes));
    }

    if (SDL_MUSTLOCK(surf)) SDL_UnlockSurface(surf);
    SDL_FreeSurface(surf);

    return out;
}

torch::Tensor render_text_utf16_rgba_f32(TTF_Font* font, const std::u16string& utf16, SdlColor fg, int wrap_width_px) {
    auto u8 = render_text_utf16_rgba_u8(font, utf16, fg, wrap_width_px);
    if (u8.numel() == 0) return u8.to(torch::kFloat32);
    return u8.to(torch::kFloat32).div_(255.0f);
}

#else

torch::Tensor render_text_utf16_rgba_u8(TTF_Font* /*font*/, const std::u16string& /*utf16*/, SdlColor /*fg*/, int /*wrap_width_px*/) {
    throw std::runtime_error("SDL_ttf not available at compile time");
}

torch::Tensor render_text_utf16_rgba_f32(TTF_Font* /*font*/, const std::u16string& /*utf16*/, SdlColor /*fg*/, int /*wrap_width_px*/) {
    throw std::runtime_error("SDL_ttf not available at compile time");
}

#endif
