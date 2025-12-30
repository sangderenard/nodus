#include "sdlttf_to_torch.h"
#include <iostream>

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    try {
        SdlTtfGuard guard;
        FontHandle f("assets/fonts/NotoSans-Regular.ttf", 24);
        SdlColor white{255,255,255,255};
        std::u16string txt = u"Hello — مرحبا — 你好";
        auto t = render_text_utf16_rgba_u8(f.font, txt, white);
        std::cout << "Rendered tensor sizes: ";
        for (auto s : t.sizes()) std::cout << s << " ";
        std::cout << " dtype=" << t.dtype() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
