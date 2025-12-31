#include "../tools/sdlttf_to_torch.h"
#if defined(__has_include)
# if __has_include(<torch/torch.h>)
#  include <torch/torch.h>
# endif
#endif

SdlTtfGuard::SdlTtfGuard() {}
SdlTtfGuard::~SdlTtfGuard() {}

FontHandle::FontHandle(const char* /*path*/, int /*pt_size*/) { throw std::runtime_error("SDL_ttf not available at link time"); }
FontHandle::~FontHandle() {}

torch::Tensor render_text_utf16_rgba_u8(TTF_Font* /*font*/, const std::u16string& /*utf16*/, SdlColor /*fg*/, int /*wrap_width_px*/) {
    throw std::runtime_error("SDL_ttf render not available at link time");
}

torch::Tensor render_text_utf16_rgba_f32(TTF_Font* /*font*/, const std::u16string& /*utf16*/, SdlColor /*fg*/, int /*wrap_width_px*/) {
    throw std::runtime_error("SDL_ttf render not available at link time");
}
