#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <fstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <SDL.h>
#include <SDL_opengl.h>
#include <torch/torch.h>
#include <filesystem>
#include <system_error>

#include "canvas_abi.h"

namespace {

struct CanvasHead {
    GP_CanvasContext* context = nullptr;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
    std::vector<uint8_t> upload_buffer;
    std::string workspace_path;
};

static constexpr size_t kMaxCanvasHeads = 8;
static constexpr const char* kDefaultCanvasWorkspacePath = "canvas_workspace.txt";
static constexpr const char* kFrontendCanvasStatePath = "frontend_canvases.txt";
static constexpr double kCanvasAutosaveIntervalSeconds = 2.0;

struct CanvasCollectionState {
    std::array<std::string, kMaxCanvasHeads> workspace_paths;
    std::array<bool, kMaxCanvasHeads> has_history{};
    size_t active_index = 0;
};
static std::string canvas_workspace_path_for_index(size_t index);
static void initialize_collection_state(CanvasCollectionState& state);
static bool load_frontend_collection_state(CanvasCollectionState& state);
static void save_frontend_collection_state(const CanvasCollectionState& state);
static constexpr int kDefaultCanvasWidth = 1000;
static constexpr int kDefaultCanvasHeight = 720;
static constexpr int kCanvasSwitchWindowMs = 3000;
struct FrontendResources {
    SDL_Window* window = nullptr;
    SDL_GLContext gl_context = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* canvas_texture = nullptr;
    int canvas_texture_width = 0;
    int canvas_texture_height = 0;
    torch::Device torch_device = torch::kCPU;
    torch::Tensor torch_handle;
    std::array<CanvasHead, kMaxCanvasHeads> canvas_heads{};
    size_t active_canvas_index = 0;
    CanvasCollectionState collection_state{};
    int canvas_width_hint = kDefaultCanvasWidth;
    int canvas_height_hint = kDefaultCanvasHeight;
    // layout info updated on window resize and used for rendering/input translation
    struct FrontendLayout {
        int window_w = 0;
        int window_h = 0;
        int canvas_w = 0;   // display width for the canvas texture
        int canvas_h = 0;   // display height for the canvas texture
        int canvas_x = 0;   // top-left x where canvas is rendered (window coords)
        int canvas_y = 0;   // top-left y where canvas is rendered (window coords)
        int left_space = 0; // positive: pixels between window left and canvas left; negative: canvas extends left
        int right_space = 0; // positive: pixels between canvas right and window right
        int bottom_space = 0; // positive: pixels between canvas bottom and window bottom
    } layout;
};

struct CanvasTickSnapshot {
    std::array<GP_CanvasContext*, kMaxCanvasHeads> contexts{};
    std::array<size_t, kMaxCanvasHeads> head_indices{};
    size_t count = 0;
};

enum class DisplayMode {
    Interval = 0,
    Ratio = 1,
};

struct DisplayState {
    std::atomic<DisplayMode> mode{DisplayMode::Interval};
    std::atomic<double> interval{1.0 / 30.0};
    std::atomic<double> ratio{1.0};
    std::atomic<int64_t> ticks_since_display{0};
    std::atomic<double> last_display_time{0.0};
};

class CanvasTickController {
public:
    explicit CanvasTickController(FrontendResources& resources)
        : resources_(resources) {
        for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
            head_thread_dt_[i].store(1.0 / 60.0);
            head_thread_ratio_[i].store(1.0);
            head_thread_free_spin_[i].store(false);
            head_gui_dt_[i].store(1.0 / 60.0);
            render_frame_counters_[i].store(0);
            display_params_[i].mode.store(DisplayMode::Interval);
            display_params_[i].interval.store(1.0 / 30.0);
            display_params_[i].ratio.store(1.0);
            display_params_[i].ticks_since_display.store(0);
            display_params_[i].last_display_time.store(0.0);
        }
        gui_delay_s_.store(1.0 / 60.0);
        thread_delay_s_.store(1.0 / 60.0);
    }

    ~CanvasTickController() { stop(); }

    void start() {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) return;
        worker_ = std::thread(&CanvasTickController::thread_loop, this);
    }

    void stop() {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false)) return;
        if (worker_.joinable()) worker_.join();
    }

    void record_render(float dt) {
        publish_snapshot();
        for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
            if (!resources_.canvas_heads[i].context) continue;
            head_gui_dt_[i].store(dt > 0.0f ? dt : (1.0 / 60.0), std::memory_order_relaxed);
            render_frame_counters_[i].fetch_add(1, std::memory_order_relaxed);
        }
    }

    void set_global_gui_delay(double seconds) {
        gui_delay_s_.store(std::max(0.0, seconds), std::memory_order_relaxed);
    }

    void set_global_thread_delay(double seconds) {
        thread_delay_s_.store(std::max(0.0, seconds), std::memory_order_relaxed);
    }

    void set_canvas_thread_config(size_t head_idx, double dt, double ratio, bool free_spin) {
        if (head_idx >= kMaxCanvasHeads) return;
        head_thread_dt_[head_idx].store(std::max(0.0, dt), std::memory_order_relaxed);
        head_thread_ratio_[head_idx].store(std::max(0.0, ratio), std::memory_order_relaxed);
        head_thread_free_spin_[head_idx].store(free_spin, std::memory_order_relaxed);
    }

    void set_canvas_gui_dt(size_t head_idx, double dt) {
        if (head_idx >= kMaxCanvasHeads) return;
        head_gui_dt_[head_idx].store(std::max(0.0, dt), std::memory_order_relaxed);
    }

    double gui_delay_seconds() const {
        return gui_delay_s_.load(std::memory_order_relaxed);
    }

    enum class DisplayMode {
        Interval = 0,
        Ratio = 1,
    };

    void set_canvas_display_interval(size_t head_idx, double interval) {
        if (head_idx >= kMaxCanvasHeads) return;
        display_params_[head_idx].mode.store(DisplayMode::Interval, std::memory_order_relaxed);
        display_params_[head_idx].interval.store(std::max(0.0, interval), std::memory_order_relaxed);
    }

    void set_canvas_display_ratio(size_t head_idx, double ratio) {
        if (head_idx >= kMaxCanvasHeads) return;
        display_params_[head_idx].mode.store(DisplayMode::Ratio, std::memory_order_relaxed);
        display_params_[head_idx].ratio.store(std::max(1.0, ratio), std::memory_order_relaxed);
    }

    bool should_display(size_t head_idx, double now_seconds) const {
        if (head_idx >= kMaxCanvasHeads) return false;
        const DisplayState& state = display_params_[head_idx];
        DisplayMode mode = state.mode.load(std::memory_order_relaxed);
        if (mode == DisplayMode::Interval) {
            double interval = state.interval.load(std::memory_order_relaxed);
            double last = state.last_display_time.load(std::memory_order_relaxed);
            return interval <= 0.0 || (now_seconds - last >= interval);
        }
        double ratio = state.ratio.load(std::memory_order_relaxed);
        double ticks = static_cast<double>(state.ticks_since_display.load(std::memory_order_relaxed));
        return ticks >= ratio;
    }

    void mark_displayed(size_t head_idx, double now_seconds) {
        if (head_idx >= kMaxCanvasHeads) return;
        auto& state = display_params_[head_idx];
        state.last_display_time.store(now_seconds, std::memory_order_relaxed);
        state.ticks_since_display.store(0, std::memory_order_relaxed);
    }

private:
    struct DisplayState {
        std::atomic<DisplayMode> mode{DisplayMode::Interval};
        std::atomic<double> interval{1.0 / 30.0};
        std::atomic<double> ratio{1.0};
        std::atomic<int64_t> ticks_since_display{0};
        std::atomic<double> last_display_time{0.0};
    };

    void publish_snapshot() {
        CanvasTickSnapshot temp{};
        for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
            if (!resources_.canvas_heads[i].context) continue;
            temp.contexts[temp.count] = resources_.canvas_heads[i].context;
            temp.head_indices[temp.count] = i;
            ++temp.count;
        }
        int next = 1 - snapshot_index_.load(std::memory_order_relaxed);
        snapshots_[next] = temp;
        snapshot_index_.store(next, std::memory_order_release);
    }

    CanvasTickSnapshot load_snapshot() const {
        return snapshots_[snapshot_index_.load(std::memory_order_acquire)];
    }

    void thread_loop() {
        std::array<double, kMaxCanvasHeads> accum{};
        std::array<uint64_t, kMaxCanvasHeads> last_frame_counts{};
        while (running_.load(std::memory_order_relaxed)) {
            CanvasTickSnapshot snapshot = load_snapshot();
            for (size_t entry = 0; entry < snapshot.count; ++entry) {
                size_t head_idx = snapshot.head_indices[entry];
                GP_CanvasContext* ctx = snapshot.contexts[entry];
                if (!ctx) continue;
                bool free_spin = head_thread_free_spin_[head_idx].load(std::memory_order_relaxed);
                double ratio = head_thread_ratio_[head_idx].load(std::memory_order_relaxed);
                double ticks_to_run = 0.0;
                if (free_spin) {
                    ticks_to_run = 1.0;
                } else {
                    uint64_t current = render_frame_counters_[head_idx].load(std::memory_order_relaxed);
                    uint64_t frames = current - last_frame_counts[head_idx];
                    if (frames == 0) continue;
                    accum[head_idx] += static_cast<double>(frames) * ratio;
                    ticks_to_run = std::floor(accum[head_idx]);
                    accum[head_idx] -= ticks_to_run;
                    last_frame_counts[head_idx] = current;
                    if (ticks_to_run <= 0.0) continue;
                }
                int paused = 1;
                if (gp_canvas_get_thread_manager_paused(ctx, &paused) && paused) continue;
                float dt = static_cast<float>(head_thread_dt_[head_idx].load(std::memory_order_relaxed));
                int tick_count = static_cast<int>(ticks_to_run);
                for (int t = 0; t < tick_count; ++t) {
                    gp_canvas_step(ctx, dt);
                }
                if (tick_count > 0) {
                    display_params_[head_idx].ticks_since_display.fetch_add(tick_count, std::memory_order_relaxed);
                }
            }
            double delay = thread_delay_s_.load(std::memory_order_relaxed);
            if (delay > 0.0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(delay));
            }
        }
    }

    FrontendResources& resources_;
    std::array<CanvasTickSnapshot, 2> snapshots_{};
    std::atomic<int> snapshot_index_{0};
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::array<std::atomic<double>, kMaxCanvasHeads> head_thread_dt_{};
    std::array<std::atomic<double>, kMaxCanvasHeads> head_thread_ratio_{};
    std::array<std::atomic<bool>, kMaxCanvasHeads> head_thread_free_spin_{};
    std::array<std::atomic<double>, kMaxCanvasHeads> head_gui_dt_{};
    std::array<std::atomic<uint64_t>, kMaxCanvasHeads> render_frame_counters_{};
    std::array<DisplayState, kMaxCanvasHeads> display_params_{};
    std::atomic<double> gui_delay_s_{1.0 / 60.0};
    std::atomic<double> thread_delay_s_{1.0 / 60.0};
};

bool create_canvas_head(CanvasHead& head, int width, int height) {
    if (head.context) {
        return true;
    }
    head.context = gp_canvas_create(width, height);
    if (!head.context) {
        return false;
    }
    head.width = width;
    head.height = height;
    head.pixels.assign(static_cast<size_t>(width) * height * 4, 0u);
    return true;
}

void destroy_canvas_head(CanvasHead& head) {
    if (!head.context) {
        return;
    }
    gp_canvas_destroy(head.context);
    head.context = nullptr;
    head.pixels.clear();
    head.width = 0;
    head.height = 0;
    head.workspace_path.clear();
}

bool ensure_canvas_head_initialized(FrontendResources& resources, size_t index) {
    if (index >= kMaxCanvasHeads) return false;
    auto& head = resources.canvas_heads[index];
    if (head.context) return true;
    if (!create_canvas_head(head, resources.canvas_width_hint, resources.canvas_height_hint)) {
        return false;
    }
    std::string workspace_path = resources.collection_state.workspace_paths[index];
    if (workspace_path.empty()) {
        workspace_path = canvas_workspace_path_for_index(index);
    }
    head.workspace_path = workspace_path;
    resources.collection_state.workspace_paths[index] = workspace_path;
    if (!workspace_path.empty()) {
        gp_canvas_set_autosave(head.context, workspace_path.c_str(), kCanvasAutosaveIntervalSeconds);
        std::ifstream ifs(workspace_path);
        if (ifs.good()) {
            if (!gp_canvas_load_from_file(head.context, workspace_path.c_str())) {
                std::cerr << "Failed to load canvas workspace: " << workspace_path << "\n";
            }
        }
    }
    resources.canvas_width_hint = head.width;
    resources.canvas_height_hint = head.height;
    gp_canvas_set_thread_manager_paused(head.context, 0);
    resources.collection_state.has_history[index] = true;
    return true;
}

void destroy_canvas_heads(FrontendResources& resources) {
    for (auto& head : resources.canvas_heads) {
        destroy_canvas_head(head);
    }
}

bool initialize_canvas_heads(FrontendResources& resources) {
    size_t target = resources.collection_state.active_index;
    if (target >= kMaxCanvasHeads) {
        target = 0;
    }
    if (!ensure_canvas_head_initialized(resources, target)) {
        std::cerr << "Failed to create initial canvas head\n";
        return false;
    }
    resources.active_canvas_index = target;
    resources.collection_state.active_index = target;
    return true;
}

bool ensure_canvas_texture(FrontendResources& resources, int width, int height) {
    if (!resources.renderer || width <= 0 || height <= 0) return false;
    if (resources.canvas_texture &&
        width == resources.canvas_texture_width &&
        height == resources.canvas_texture_height) {
        return true;
    }
    if (resources.canvas_texture) {
        SDL_DestroyTexture(resources.canvas_texture);
        resources.canvas_texture = nullptr;
    }
    resources.canvas_texture = SDL_CreateTexture(
        resources.renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height
    );
    if (!resources.canvas_texture) return false;
    // Log the actual pixel format SDL created so we can verify byte layout expectations.
    Uint32 fmt = 0; int access = 0; int w = 0; int h = 0;
    SDL_QueryTexture(resources.canvas_texture, &fmt, &access, &w, &h);
    std::cerr << "Canvas texture format: " << SDL_GetPixelFormatName(fmt) << "\n";
    resources.canvas_texture_width = width;
    resources.canvas_texture_height = height;
    return true;
}

bool update_active_canvas_texture(FrontendResources& resources) {
    size_t index = resources.active_canvas_index;
    if (index >= resources.canvas_heads.size()) return false;
    auto& head = resources.canvas_heads[index];
    if (!head.context || head.width <= 0 || head.height <= 0) return false;
    if (!ensure_canvas_texture(resources, head.width, head.height)) return false;
    size_t buf_len = static_cast<size_t>(head.width) * head.height * 4u;
    if (head.pixels.size() < buf_len) head.pixels.resize(buf_len);
    if (!gp_canvas_raster_rgba(head.context, head.pixels.data(), static_cast<int32_t>(buf_len))) {
        return false;
    }
    int pitch = head.width * 4;
    SDL_UpdateTexture(resources.canvas_texture, nullptr, head.pixels.data(), pitch);
    return true;
}

bool initialize_window(FrontendResources& resources) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return false;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    resources.window = SDL_CreateWindow(
        "Canvas Frontend Shell",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        1280,
        720,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
    );

    if (!resources.window) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        return false;
    }

    resources.gl_context = SDL_GL_CreateContext(resources.window);
    if (!resources.gl_context) {
        std::cerr << "SDL_GL_CreateContext failed: " << SDL_GetError() << "\n";
        return false;
    }

    if (SDL_GL_MakeCurrent(resources.window, resources.gl_context) != 0) {
        std::cerr << "SDL_GL_MakeCurrent failed: " << SDL_GetError() << "\n";
        return false;
    }

    SDL_GL_SetSwapInterval(1);
    resources.renderer = SDL_CreateRenderer(resources.window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!resources.renderer) {
        std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << "\n";
        return false;
    }
    return true;
}

void cleanup(FrontendResources& resources) {
    destroy_canvas_heads(resources);
    if (resources.canvas_texture) {
        SDL_DestroyTexture(resources.canvas_texture);
        resources.canvas_texture = nullptr;
    }
    if (resources.renderer) {
        SDL_DestroyRenderer(resources.renderer);
        resources.renderer = nullptr;
    }
    if (resources.gl_context) {
        SDL_GL_DeleteContext(resources.gl_context);
    }
    if (resources.window) {
        SDL_DestroyWindow(resources.window);
    }
    SDL_Quit();
}

struct InputFilters {
    bool keyboard_enabled = true;
    bool mouse_enabled = true;
    bool gamepad_enabled = false;
    bool keyboard_filter_active = false;
    std::unordered_set<SDL_Scancode> keyboard_keys;
    bool mouse_filter_active = false;
    std::unordered_set<Uint8> mouse_buttons;
    bool gamepad_filter_active = false;
    std::unordered_set<SDL_GameControllerButton> gamepad_buttons;
};

struct CanvasSwitchCombo {
    bool hunting = false;
    std::chrono::steady_clock::time_point start_time{};
    std::string digits;
    bool clone_from_active = false;
};

struct GamepadInfo {
    SDL_GameController* controller = nullptr;
    SDL_JoystickID instance_id = -1;
};

class InputDispatcher {
public:
    InputDispatcher(FrontendResources& resources, InputFilters filters);
    InputDispatcher(const InputDispatcher&) = delete;
    InputDispatcher& operator=(const InputDispatcher&) = delete;
    ~InputDispatcher();

    void handle_event(const SDL_Event& event);
    void update();
    // notify dispatcher that window size changed (window coords)
    void on_window_resized(int window_w, int window_h);

private:
    bool should_forward_keyboard(const SDL_KeyboardEvent& event) const;
    bool should_forward_mouse_button(const SDL_MouseButtonEvent& event) const;
    bool handle_combo_key(const SDL_KeyboardEvent& event);
    void maybe_complete_combo();
    void start_combo(bool clone_requested = false);
    void complete_combo();
    int scancode_to_digit(SDL_Scancode scan) const;
    void switch_to_canvas(size_t index);
    void add_gamepad(int device_index);
    void remove_gamepad(SDL_JoystickID instance_id);
    void update_modifier_state(const SDL_KeyboardEvent& event, bool pressed);

    FrontendResources& resources_;
    InputFilters filters_;
    std::vector<GamepadInfo> gamepads_;
    CanvasSwitchCombo combo_;
    bool shift_down_ = false;
    bool ctrl_down_ = false;
    bool alt_down_ = false;
};

InputDispatcher::InputDispatcher(FrontendResources& resources, InputFilters filters)
    : resources_(resources), filters_(std::move(filters)) {}

InputDispatcher::~InputDispatcher() {
    for (auto& info : gamepads_) {
        if (info.controller) {
            SDL_GameControllerClose(info.controller);
            info.controller = nullptr;
        }
    }
}

void InputDispatcher::handle_event(const SDL_Event& event) {
    maybe_complete_combo();
    switch (event.type) {
        case SDL_KEYDOWN:
        case SDL_KEYUP: {
            update_modifier_state(event.key, event.type == SDL_KEYDOWN);
            if (handle_combo_key(event.key)) {
                return;
            }
            if (!filters_.keyboard_enabled) break;
            if (!should_forward_keyboard(event.key)) break;
            auto* ctx = resources_.canvas_heads[resources_.active_canvas_index].context;
            if (!ctx) break;
            gp_canvas_on_key(
                ctx,
                event.key.keysym.sym,
                event.key.keysym.scancode,
                event.type == SDL_KEYDOWN ? 1 : 0,
                event.key.keysym.mod
            );
            break;
        }
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP: {
            if (!filters_.mouse_enabled) break;
            if (!should_forward_mouse_button(event.button)) break;
            auto* ctx = resources_.canvas_heads[resources_.active_canvas_index].context;
            if (!ctx) break;
            // translate window coords -> canvas-local coords using layout
            int lx = event.button.x - resources_.layout.canvas_x;
            int ly = event.button.y - resources_.layout.canvas_y;
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                gp_canvas_on_mouse_down(ctx, lx, ly);
            } else {
                gp_canvas_on_mouse_up(ctx, lx, ly);
            }
            break;
        }
        case SDL_MOUSEMOTION: {
            if (!filters_.mouse_enabled) break;
            auto* ctx = resources_.canvas_heads[resources_.active_canvas_index].context;
            if (!ctx) break;
            int lx = event.motion.x - resources_.layout.canvas_x;
            int ly = event.motion.y - resources_.layout.canvas_y;
            gp_canvas_on_mouse_move(ctx, lx, ly);
            break;
        }
        case SDL_CONTROLLERDEVICEADDED:
            add_gamepad(event.cdevice.which);
            break;
        case SDL_CONTROLLERDEVICEREMOVED:
            remove_gamepad(event.cdevice.which);
            break;
        case SDL_CONTROLLERBUTTONDOWN:
        case SDL_CONTROLLERBUTTONUP:
            if (!filters_.gamepad_enabled) break;
            if (!filters_.gamepad_filter_active ||
                filters_.gamepad_buttons.count(static_cast<SDL_GameControllerButton>(event.cbutton.button)) > 0) {
                if (filters_.gamepad_filter_active) {
                    std::cout << "Gamepad button "
                              << static_cast<int>(event.cbutton.button)
                              << (event.type == SDL_CONTROLLERBUTTONDOWN ? " down" : " up")
                              << "\n";
                }
            }
            break;
        default:
            break;
    }
}

void InputDispatcher::update_modifier_state(const SDL_KeyboardEvent& event, bool pressed) {
    switch (event.keysym.scancode) {
        case SDL_SCANCODE_LSHIFT:
        case SDL_SCANCODE_RSHIFT:
            shift_down_ = pressed;
            break;
        case SDL_SCANCODE_LCTRL:
        case SDL_SCANCODE_RCTRL:
            ctrl_down_ = pressed;
            break;
        case SDL_SCANCODE_LALT:
        case SDL_SCANCODE_RALT:
            alt_down_ = pressed;
            break;
        default:
            break;
    }
}

bool InputDispatcher::should_forward_keyboard(const SDL_KeyboardEvent& event) const {
    if (!filters_.keyboard_filter_active) return true;
    return filters_.keyboard_keys.count(event.keysym.scancode) > 0;
}

bool InputDispatcher::should_forward_mouse_button(const SDL_MouseButtonEvent& event) const {
    if (!filters_.mouse_filter_active) return true;
    return filters_.mouse_buttons.count(event.button) > 0;
}

void InputDispatcher::update() {
    maybe_complete_combo();
}

void InputDispatcher::on_window_resized(int window_w, int window_h) {
    auto& layout = resources_.layout;
    layout.window_w = window_w;
    layout.window_h = window_h;
    int cw = resources_.canvas_texture_width > 0 ? resources_.canvas_texture_width : resources_.canvas_width_hint;
    int ch = resources_.canvas_texture_height > 0 ? resources_.canvas_texture_height : resources_.canvas_height_hint;
    layout.canvas_w = cw;
    layout.canvas_h = ch;
    layout.canvas_x = (window_w - cw) / 2;
    layout.canvas_y = 0;
    layout.left_space = layout.canvas_x;
    layout.right_space = window_w - (layout.canvas_x + layout.canvas_w);
    layout.bottom_space = window_h - (layout.canvas_y + layout.canvas_h);

    // Update each canvas head and notify canvas contexts of new logical size
    for (auto& head : resources_.canvas_heads) {
        head.width = cw;
        head.height = ch;
        head.pixels.assign(static_cast<size_t>(std::max(1, cw)) * std::max(1, ch) * 4, 0u);
        if (head.context) {
            gp_canvas_set_size(head.context, cw, ch);
        }
    }
    // Ensure the SDL texture matches the chosen canvas size
    ensure_canvas_texture(resources_, cw, ch);
}

void InputDispatcher::maybe_complete_combo() {
    if (!combo_.hunting) return;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - combo_.start_time).count();
    if (elapsed >= kCanvasSwitchWindowMs) {
        complete_combo();
    }
}

bool InputDispatcher::handle_combo_key(const SDL_KeyboardEvent& event) {
    if (event.type != SDL_KEYDOWN) return false;
    int digit = scancode_to_digit(event.keysym.scancode);
    if (event.keysym.scancode == SDL_SCANCODE_TAB &&
        shift_down_ && ctrl_down_) {
        if (!combo_.hunting) {
            start_combo(alt_down_);
        } else {
            combo_.start_time = std::chrono::steady_clock::now();
            combo_.clone_from_active = alt_down_;
        }
        return true;
    }
    if (combo_.hunting && digit >= 0) {
        if (combo_.digits.size() < 3) {
            combo_.digits.push_back(static_cast<char>('0' + digit));
            std::cout << "Canvas hunt saw digit " << digit << " (scancode " << event.keysym.scancode << " mods=" << event.keysym.mod << ")\n";
        }
        return true;
    }
    if (combo_.hunting) {
        std::cout << "Canvas hunt ignored key sym=" << event.keysym.sym << " sc=" << event.keysym.scancode << " mod=" << event.keysym.mod << "\n";
    }
    return false;
}

void InputDispatcher::start_combo(bool clone_requested) {
    combo_.hunting = true;
    combo_.start_time = std::chrono::steady_clock::now();
    combo_.digits.clear();
    combo_.clone_from_active = clone_requested;
    std::cout << "Canvas hunt started\n";
}

void InputDispatcher::complete_combo() {
    if (!combo_.hunting) return;
    combo_.hunting = false;
    bool clone_requested = combo_.clone_from_active;
    combo_.clone_from_active = false;
    if (combo_.digits.empty()) return;
    int parsed = 0;
    for (char ch : combo_.digits) {
        parsed = parsed * 10 + (ch - '0');
        if (parsed > static_cast<int>(kMaxCanvasHeads)) {
            break;
        }
    }
    combo_.digits.clear();
    if (parsed <= 0 || parsed > static_cast<int>(kMaxCanvasHeads)) {
        return;
    }
    std::cout << "Canvas hunt complete target=" << parsed << "\n";
    size_t target_index = static_cast<size_t>(parsed - 1);
    if (clone_requested &&
        target_index < kMaxCanvasHeads &&
        target_index != resources_.active_canvas_index) {
        const auto& collection = resources_.collection_state;
        const std::string& source_path = collection.workspace_paths[collection.active_index];
        const std::string& target_path = collection.workspace_paths[target_index];
        if (!source_path.empty() && !target_path.empty()) {
            std::error_code ec;
            if (std::filesystem::exists(source_path, ec) && !ec) {
                std::filesystem::copy_file(source_path, target_path,
                    std::filesystem::copy_options::overwrite_existing, ec);
                if (!ec) {
                    resources_.collection_state.has_history[target_index] = true;
                    std::cout << "Cloned canvas " << collection.active_index
                              << " -> " << target_index << "\n";
                } else {
                    std::cerr << "Failed to clone canvas workspace: " << ec.message() << "\n";
                }
            }
        }
    }
    switch_to_canvas(target_index);
}

int InputDispatcher::scancode_to_digit(SDL_Scancode scan) const {
    switch (scan) {
        case SDL_SCANCODE_0: case SDL_SCANCODE_KP_0: return 0;
        case SDL_SCANCODE_1: case SDL_SCANCODE_KP_1: return 1;
        case SDL_SCANCODE_2: case SDL_SCANCODE_KP_2: return 2;
        case SDL_SCANCODE_3: case SDL_SCANCODE_KP_3: return 3;
        case SDL_SCANCODE_4: case SDL_SCANCODE_KP_4: return 4;
        case SDL_SCANCODE_5: case SDL_SCANCODE_KP_5: return 5;
        case SDL_SCANCODE_6: case SDL_SCANCODE_KP_6: return 6;
        case SDL_SCANCODE_7: case SDL_SCANCODE_KP_7: return 7;
        case SDL_SCANCODE_8: case SDL_SCANCODE_KP_8: return 8;
        case SDL_SCANCODE_9: case SDL_SCANCODE_KP_9: return 9;
        default: return -1;
    }
}

void InputDispatcher::switch_to_canvas(size_t index) {
    if (index >= kMaxCanvasHeads) return;
    if (!ensure_canvas_head_initialized(resources_, index)) {
        std::cerr << "Failed to prepare canvas head " << index << "\n";
        return;
    }
    resources_.active_canvas_index = index;
    resources_.collection_state.active_index = index;
    std::cout << "Switched to canvas head " << index << "\n";
}

void InputDispatcher::add_gamepad(int device_index) {
    SDL_GameController* controller = SDL_GameControllerOpen(device_index);
    if (!controller) {
        std::cerr << "Gamepad open failed: " << SDL_GetError() << "\n";
        return;
    }
    SDL_Joystick* joy = SDL_GameControllerGetJoystick(controller);
    if (!joy) {
        SDL_GameControllerClose(controller);
        return;
    }
    SDL_JoystickID instance_id = SDL_JoystickInstanceID(joy);
    if (instance_id < 0) {
        SDL_GameControllerClose(controller);
        return;
    }
    for (const auto& info : gamepads_) {
        if (info.instance_id == instance_id) {
            SDL_GameControllerClose(controller);
            return;
        }
    }
    gamepads_.push_back({controller, instance_id});
    std::cout << "Gamepad connected (instance " << instance_id << ")\n";
}

void InputDispatcher::remove_gamepad(SDL_JoystickID instance_id) {
    auto it = std::find_if(gamepads_.begin(), gamepads_.end(),
        [instance_id](const GamepadInfo& info) { return info.instance_id == instance_id; });
    if (it == gamepads_.end()) return;
    if (it->controller) {
        SDL_GameControllerClose(it->controller);
    }
    std::cout << "Gamepad disconnected (instance " << instance_id << ")\n";
    gamepads_.erase(it);
}

static std::string trim(std::string value) {
    auto begin = std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); });
    auto end = std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base();
    if (begin < end) {
        value = std::string(begin, end);
    } else {
        value.clear();
    }
    return value;
}

static std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static std::vector<std::string> split_comma_list(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, ',')) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

static bool parse_positive_int(const std::string& text, int& value) {
    value = 0;
    bool seen = false;
    for (char ch : text) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
        seen = true;
        value = value * 10 + (ch - '0');
    }
    return seen;
}

static std::optional<Uint8> mouse_button_from_token(const std::string& raw) {
    static const std::unordered_map<std::string, Uint8> kMouseButtonMap = {
        {"left", SDL_BUTTON_LEFT},
        {"right", SDL_BUTTON_RIGHT},
        {"middle", SDL_BUTTON_MIDDLE},
        {"x1", SDL_BUTTON_X1},
        {"x2", SDL_BUTTON_X2},
    };
    auto lower = to_lower(raw);
    if (auto it = kMouseButtonMap.find(lower); it != kMouseButtonMap.end()) {
        return it->second;
    }
    if (lower.rfind("button", 0) == 0) {
        int parsed = 0;
        if (parse_positive_int(lower.substr(6), parsed)) {
            if (parsed > 0 && parsed <= 8) {
                return SDL_BUTTON(parsed);
            }
        }
    }
    int parsed = 0;
    if (parse_positive_int(lower, parsed) && parsed > 0 && parsed <= 8) {
        return SDL_BUTTON(parsed);
    }
    return std::nullopt;
}

static std::optional<SDL_GameControllerButton> gamepad_button_from_token(const std::string& raw) {
    static const std::unordered_map<std::string, SDL_GameControllerButton> kGamepadButtonMap = {
        {"a", SDL_CONTROLLER_BUTTON_A},
        {"b", SDL_CONTROLLER_BUTTON_B},
        {"x", SDL_CONTROLLER_BUTTON_X},
        {"y", SDL_CONTROLLER_BUTTON_Y},
        {"back", SDL_CONTROLLER_BUTTON_BACK},
        {"guide", SDL_CONTROLLER_BUTTON_GUIDE},
        {"start", SDL_CONTROLLER_BUTTON_START},
        {"leftstick", SDL_CONTROLLER_BUTTON_LEFTSTICK},
        {"rightstick", SDL_CONTROLLER_BUTTON_RIGHTSTICK},
        {"leftshoulder", SDL_CONTROLLER_BUTTON_LEFTSHOULDER},
        {"rightshoulder", SDL_CONTROLLER_BUTTON_RIGHTSHOULDER},
        {"dpup", SDL_CONTROLLER_BUTTON_DPAD_UP},
        {"dpdown", SDL_CONTROLLER_BUTTON_DPAD_DOWN},
        {"dpleft", SDL_CONTROLLER_BUTTON_DPAD_LEFT},
        {"dpright", SDL_CONTROLLER_BUTTON_DPAD_RIGHT},
        {"misc1", SDL_CONTROLLER_BUTTON_MISC1},
        {"paddle1", SDL_CONTROLLER_BUTTON_PADDLE1},
        {"paddle2", SDL_CONTROLLER_BUTTON_PADDLE2},
        {"paddle3", SDL_CONTROLLER_BUTTON_PADDLE3},
        {"paddle4", SDL_CONTROLLER_BUTTON_PADDLE4},
        {"touchpad", SDL_CONTROLLER_BUTTON_TOUCHPAD},
    };
    auto lower = to_lower(raw);
    if (auto it = kGamepadButtonMap.find(lower); it != kGamepadButtonMap.end()) {
        return it->second;
    }
    return std::nullopt;
}

static void apply_keyboard_filter(InputFilters& filters, const std::string& raw_value) {
    auto tokens = split_comma_list(raw_value);
    if (tokens.empty()) return;
    for (const auto& token : tokens) {
        if (token.empty()) continue;
        auto normalized = to_lower(token);
        if (normalized == "any" || normalized == "all") {
            filters.keyboard_filter_active = false;
            filters.keyboard_keys.clear();
            return;
        }
        if (normalized == "none") {
            filters.keyboard_filter_active = true;
            filters.keyboard_keys.clear();
            continue;
        }
        SDL_Scancode code = SDL_GetScancodeFromName(token.c_str());
        if (code == SDL_SCANCODE_UNKNOWN) {
            std::cerr << "Unknown keyboard scancode '" << token << "'\n";
            continue;
        }
        filters.keyboard_filter_active = true;
        filters.keyboard_keys.insert(code);
    }
}

static void apply_mouse_filter(InputFilters& filters, const std::string& raw_value) {
    auto tokens = split_comma_list(raw_value);
    if (tokens.empty()) return;
    for (const auto& token : tokens) {
        if (token.empty()) continue;
        auto normalized = to_lower(token);
        if (normalized == "any" || normalized == "all") {
            filters.mouse_filter_active = false;
            filters.mouse_buttons.clear();
            return;
        }
        if (normalized == "none") {
            filters.mouse_filter_active = true;
            filters.mouse_buttons.clear();
            continue;
        }
        if (auto button = mouse_button_from_token(token)) {
            filters.mouse_filter_active = true;
            filters.mouse_buttons.insert(*button);
        } else {
            std::cerr << "Unknown mouse button '" << token << "'\n";
        }
    }
}

static void apply_gamepad_filter(InputFilters& filters, const std::string& raw_value) {
    auto tokens = split_comma_list(raw_value);
    if (tokens.empty()) return;
    for (const auto& token : tokens) {
        if (token.empty()) continue;
        auto normalized = to_lower(token);
        if (normalized == "any" || normalized == "all") {
            filters.gamepad_filter_active = false;
            filters.gamepad_buttons.clear();
            return;
        }
        if (normalized == "none") {
            filters.gamepad_filter_active = true;
            filters.gamepad_buttons.clear();
            continue;
        }
        if (auto button = gamepad_button_from_token(token)) {
            filters.gamepad_filter_active = true;
            filters.gamepad_buttons.insert(*button);
        } else {
            std::cerr << "Unknown gamepad button '" << token << "'\n";
        }
    }
}

InputFilters parse_input_filters(int argc, char** argv) {
    InputFilters filters;
    const std::string keyboard_prefix = "--keyboard-filter=";
    const std::string mouse_prefix = "--mouse-filter=";
    const std::string gamepad_prefix = "--gamepad-filter=";
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--disable-keyboard") {
            filters.keyboard_enabled = false;
        } else if (arg == "--disable-mouse") {
            filters.mouse_enabled = false;
        } else if (arg == "--enable-gamepad") {
            filters.gamepad_enabled = true;
        } else if (arg == "--disable-gamepad") {
            filters.gamepad_enabled = false;
        } else if (arg.rfind(keyboard_prefix, 0) == 0) {
            apply_keyboard_filter(filters, arg.substr(keyboard_prefix.size()));
        } else if (arg.rfind(mouse_prefix, 0) == 0) {
            apply_mouse_filter(filters, arg.substr(mouse_prefix.size()));
        } else if (arg.rfind(gamepad_prefix, 0) == 0) {
            apply_gamepad_filter(filters, arg.substr(gamepad_prefix.size()));
        }
    }
    return filters;
}

struct FrontendOptions {
    bool show_help = false;
    bool verbose = false;
    std::string device = "auto"; // cpu, cuda, auto
    int width = kDefaultCanvasWidth;
    int height = kDefaultCanvasHeight;
    std::string workspace_path;
    // timing overrides: negative => not set
    double thread_delay = -1.0;
    double gui_delay = -1.0;

    struct CanvasConfig {
        int idx = 0;
        double dt = 1.0/60.0;
        double ratio = 1.0;
        bool free_spin = false;
        double display_interval = -1.0; // if >=0 set interval mode
        double display_ratio = -1.0; // if >=0 set ratio mode
        bool pause = false; // if true, set canvas thread manager paused
    };
    std::vector<CanvasConfig> canvas_configs;
};

static void print_usage(const char* progname) {
    std::cout << "Usage: " << progname << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help             Show this help message and exit\n";
    std::cout << "  --device=<cpu|cuda|auto>  Choose torch device (default: auto)\n";
    std::cout << "  --width=<pixels>       Initial canvas width (default: 1000)\n";
    std::cout << "  --height=<pixels>      Initial canvas height (default: 720)\n";
    std::cout << "  --workspace=<path>     Path for primary canvas workspace file\n";
    std::cout << "  --verbose              Enable verbose logging\n";
    std::cout << "  --thread-delay=<s>     Global thread loop sleep (seconds, default ~0.016)\n";
    std::cout << "  --gui-delay=<s>        Global GUI delay used by tick controller (seconds)\n";
    std::cout << "  --canvas-config=IDX:DT:RATIO:FREE  Per-canvas thread config (FREE=0/1)\n";
    std::cout << "  --canvas-display-interval=IDX:SEC  Set display interval for canvas IDX\n";
    std::cout << "  --canvas-display-ratio=IDX:RATIO   Set display ratio for canvas IDX\n";
    std::cout << "  --pause-canvas=IDX     Start canvas IDX with thread manager paused\n";
    std::cout << "  --disable-keyboard     Disable keyboard input (existing option)\n";
    std::cout << "  --disable-mouse        Disable mouse input (existing option)\n";
    std::cout << "  --enable-gamepad       Enable gamepad input (existing option)\n";
}

FrontendOptions parse_frontend_options(int argc, char** argv) {
    FrontendOptions opts;
    const std::string device_prefix = "--device=";
    const std::string width_prefix = "--width=";
    const std::string height_prefix = "--height=";
    const std::string workspace_prefix = "--workspace=";
    const std::string thread_delay_prefix = "--thread-delay=";
    const std::string gui_delay_prefix = "--gui-delay=";
    const std::string canvas_config_prefix = "--canvas-config="; // IDX:DT:RATIO:FREE
    const std::string canvas_display_interval_prefix = "--canvas-display-interval="; // IDX:SEC
    const std::string canvas_display_ratio_prefix = "--canvas-display-ratio="; // IDX:RATIO
    const std::string pause_canvas_prefix = "--pause-canvas="; // IDX
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg.rfind(device_prefix, 0) == 0) {
            opts.device = arg.substr(device_prefix.size());
        } else if (arg.rfind(width_prefix, 0) == 0) {
            opts.width = std::stoi(arg.substr(width_prefix.size()));
        } else if (arg.rfind(height_prefix, 0) == 0) {
            opts.height = std::stoi(arg.substr(height_prefix.size()));
        } else if (arg.rfind(workspace_prefix, 0) == 0) {
            opts.workspace_path = arg.substr(workspace_prefix.size());
        } else if (arg.rfind(thread_delay_prefix, 0) == 0) {
            opts.thread_delay = std::stod(arg.substr(thread_delay_prefix.size()));
        } else if (arg.rfind(gui_delay_prefix, 0) == 0) {
            opts.gui_delay = std::stod(arg.substr(gui_delay_prefix.size()));
        } else if (arg.rfind(canvas_config_prefix, 0) == 0) {
            std::string v = arg.substr(canvas_config_prefix.size());
            // expected IDX:DT:RATIO:FREE
            FrontendOptions::CanvasConfig cc;
            std::replace(v.begin(), v.end(), ':', ' ');
            std::istringstream iss(v);
            int free_i = 0;
            if ((iss >> cc.idx >> cc.dt >> cc.ratio >> free_i)) {
                cc.free_spin = (free_i != 0);
                opts.canvas_configs.push_back(cc);
            }
        } else if (arg.rfind(canvas_display_interval_prefix, 0) == 0) {
            std::string v = arg.substr(canvas_display_interval_prefix.size());
            std::replace(v.begin(), v.end(), ':', ' ');
            std::istringstream iss(v);
            FrontendOptions::CanvasConfig cc;
            if (iss >> cc.idx >> cc.display_interval) {
                opts.canvas_configs.push_back(cc);
            }
        } else if (arg.rfind(canvas_display_ratio_prefix, 0) == 0) {
            std::string v = arg.substr(canvas_display_ratio_prefix.size());
            std::replace(v.begin(), v.end(), ':', ' ');
            std::istringstream iss(v);
            FrontendOptions::CanvasConfig cc;
            if (iss >> cc.idx >> cc.display_ratio) {
                opts.canvas_configs.push_back(cc);
            }
        } else if (arg.rfind(pause_canvas_prefix, 0) == 0) {
            std::string v = arg.substr(pause_canvas_prefix.size());
            int idx = std::stoi(v);
            FrontendOptions::CanvasConfig cc; cc.idx = idx; cc.pause = true; opts.canvas_configs.push_back(cc);
        }
    }
    return opts;
}

static std::string canvas_workspace_path_for_index(size_t index) {
    if (index == 0) {
        return std::string(kDefaultCanvasWorkspacePath);
    }
    std::ostringstream ss;
    ss << "canvas_workspace_" << index << ".txt";
    return ss.str();
}

static void initialize_collection_state(CanvasCollectionState& state) {
    for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
        state.workspace_paths[i] = canvas_workspace_path_for_index(i);
        state.has_history[i] = false;
    }
    state.active_index = 0;
}

static bool load_frontend_collection_state(CanvasCollectionState& state) {
    std::ifstream ifs(kFrontendCanvasStatePath);
    if (!ifs.good()) {
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;
        if (keyword == "ACTIVE") {
            size_t idx = 0;
            if (iss >> idx && idx < kMaxCanvasHeads) {
                state.active_index = idx;
            }
        } else if (keyword == "CANVAS") {
            size_t idx = 0;
            if (!(iss >> idx) || idx >= kMaxCanvasHeads) continue;
            state.workspace_paths[idx] = canvas_workspace_path_for_index(idx);
            state.has_history[idx] = true;
        }
    }
    return true;
}

static void save_frontend_collection_state(const CanvasCollectionState& state) {
    std::ofstream ofs(kFrontendCanvasStatePath, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Unable to write frontend canvas state to " << kFrontendCanvasStatePath << "\n";
        return;
    }
    size_t active = state.active_index;
    if (active >= kMaxCanvasHeads) {
        active = 0;
    }
    ofs << "ACTIVE " << active << "\n";
    for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
        if (!state.has_history[i]) continue;
        ofs << "CANVAS " << i << " " << canvas_workspace_path_for_index(i) << "\n";
    }
}

void pump_events(bool& running, InputDispatcher& dispatcher) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT:
                if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                    running = false;
                } else if (event.window.event == SDL_WINDOWEVENT_RESIZED ||
                           event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    SDL_Window* w = SDL_GetWindowFromID(event.window.windowID);
                    if (w) {
                        int ww = 0, wh = 0;
                        SDL_GetWindowSize(w, &ww, &wh);
                        dispatcher.on_window_resized(ww, wh);
                    }
                }
                break;
            default:
                break;
        }
        dispatcher.handle_event(event);
    }
}

void render_frame(FrontendResources& resources, CanvasTickController& controller) {
    if (!resources.renderer) return;
    auto now = std::chrono::steady_clock::now();
    double now_seconds = std::chrono::duration<double>(now.time_since_epoch()).count();
    size_t active_idx = resources.active_canvas_index;
    bool should_update = controller.should_display(active_idx, now_seconds);
    if (should_update) {
        update_active_canvas_texture(resources);
        controller.mark_displayed(active_idx, now_seconds);
    }

    // Clear full window background
    SDL_SetRenderDrawColor(resources.renderer, 20, 24, 34, 255);
    SDL_RenderClear(resources.renderer);
    // Render canvas texture at computed top-centered position
    if (resources.canvas_texture) {
        SDL_Rect dst;
        dst.x = resources.layout.canvas_x;
        dst.y = resources.layout.canvas_y;
        dst.w = resources.layout.canvas_w > 0 ? resources.layout.canvas_w : resources.canvas_texture_width;
        dst.h = resources.layout.canvas_h > 0 ? resources.layout.canvas_h : resources.canvas_texture_height;
        SDL_RenderCopy(resources.renderer, resources.canvas_texture, nullptr, &dst);
    }
    SDL_RenderPresent(resources.renderer);
}

void prepare_torch(FrontendResources& resources, const FrontendOptions& opts) {
    // Decide device based on options and availability
    std::string dev = to_lower(opts.device);
    bool cuda_available = false;
    // runtime check for CUDA availability
    try {
        cuda_available = torch::cuda::is_available();
    } catch (...) {
        cuda_available = false;
    }
    if (dev == "cpu") {
        resources.torch_device = torch::Device(torch::kCPU);
    } else if (dev == "cuda") {
        if (cuda_available) {
            resources.torch_device = torch::Device(torch::kCUDA);
        } else {
            std::cerr << "Requested CUDA but CUDA not available; falling back to CPU\n";
            resources.torch_device = torch::Device(torch::kCPU);
        }
    } else { // auto or unknown -> prefer CUDA if available
        if (cuda_available) {
            resources.torch_device = torch::Device(torch::kCUDA);
        } else {
            resources.torch_device = torch::Device(torch::kCPU);
        }
    }

    resources.torch_handle = torch::zeros({64, 64}, torch::dtype(torch::kFloat32).device(resources.torch_device));

    // Apply display/workspace hints from options
    resources.canvas_width_hint = opts.width;
    resources.canvas_height_hint = opts.height;
    if (!opts.workspace_path.empty()) {
        resources.collection_state.workspace_paths[resources.collection_state.active_index] = opts.workspace_path;
    }

    if (opts.verbose) {
        std::cout << "Using Torch device: " << resources.torch_device.str() << "\n";
        std::cout << "Canvas size hint: " << resources.canvas_width_hint << "x" << resources.canvas_height_hint << "\n";
        if (!opts.workspace_path.empty()) std::cout << "Workspace: " << opts.workspace_path << "\n";
    }

    // initialize layout with hints; actual window size will update later
    resources.layout.canvas_w = resources.canvas_width_hint;
    resources.layout.canvas_h = resources.canvas_height_hint;
}

}  // namespace

int main(int argc, char** argv) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMECONTROLLER) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return EXIT_FAILURE;
    }
    SDL_GameControllerEventState(SDL_ENABLE);
    FrontendOptions options = parse_frontend_options(argc, argv);
    if (options.show_help) {
        print_usage(argv && argv[0] ? argv[0] : "frontend_shell");
        return EXIT_SUCCESS;
    }

    InputFilters input_filters = parse_input_filters(argc, argv);

    FrontendResources resources;
    initialize_collection_state(resources.collection_state);
    if (load_frontend_collection_state(resources.collection_state)) {
        std::cout << "Restored frontend canvas collection (active="
                  << resources.collection_state.active_index << ")\n";
    }
    prepare_torch(resources, options);

    if (!initialize_window(resources)) {
        cleanup(resources);
        return EXIT_FAILURE;
    }

    if (!initialize_canvas_heads(resources)) {
        cleanup(resources);
        return EXIT_FAILURE;
    }

    std::cout << "Torch backend: " << resources.torch_device.str() << "\n";

    constexpr float kCanvasStepDt = 1.0f / 60.0f;
    CanvasTickController tick_controller(resources);
    // apply CLI overrides if provided
    if (options.thread_delay >= 0.0) tick_controller.set_global_thread_delay(options.thread_delay);
    else tick_controller.set_global_thread_delay(kCanvasStepDt);
    if (options.gui_delay >= 0.0) tick_controller.set_global_gui_delay(options.gui_delay);
    else tick_controller.set_global_gui_delay(kCanvasStepDt);

    // apply per-canvas configs
    for (const auto& cc : options.canvas_configs) {
        if (cc.idx < 0 || cc.idx >= static_cast<int>(resources.canvas_heads.size())) continue;
        tick_controller.set_canvas_thread_config(static_cast<size_t>(cc.idx), cc.dt, cc.ratio, cc.free_spin);
        if (cc.display_interval >= 0.0) tick_controller.set_canvas_display_interval(static_cast<size_t>(cc.idx), cc.display_interval);
        if (cc.display_ratio >= 0.0) tick_controller.set_canvas_display_ratio(static_cast<size_t>(cc.idx), cc.display_ratio);
    }

    tick_controller.start();

    {
        InputDispatcher dispatcher(resources, input_filters);
        // initialize layout from current window size
        if (resources.window) {
            int ww = 0, wh = 0;
            SDL_GetWindowSize(resources.window, &ww, &wh);
            dispatcher.on_window_resized(ww, wh);
        }
            // apply pause flags to canvases if requested
            for (const auto& cc : options.canvas_configs) {
                if (cc.pause && cc.idx >= 0 && cc.idx < static_cast<int>(resources.canvas_heads.size())) {
                    auto& head = resources.canvas_heads[cc.idx];
                    if (head.context) gp_canvas_set_thread_manager_paused(head.context, 1);
                }
            }
            bool running = true;
        while (running) {
            pump_events(running, dispatcher);
            dispatcher.update();
            tick_controller.record_render(kCanvasStepDt);
            render_frame(resources, tick_controller);
            double gui_delay = tick_controller.gui_delay_seconds();
            if (gui_delay > 0.0) {
                Uint32 delay_ms = static_cast<Uint32>(std::max(0.0, gui_delay * 1000.0));
                SDL_Delay(delay_ms);
            } else {
                SDL_Delay(0);
            }
        }
        tick_controller.stop();
    }

    save_frontend_collection_state(resources.collection_state);

    cleanup(resources);
    return EXIT_SUCCESS;
}
