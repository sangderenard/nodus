// Loads a real bridge-generated canvas (turing's nodus_canvas_kpn.py output)
// through the headless ABI -- no window, no renderer, no GUI crash surface --
// pushes known values onto every genuinely external/boundary input contact,
// runs the real ThreadManager scheduler to quiescence, and pulls the computed
// result back out. The expected value is an independent Python-side oracle
// (plain math, zero nodus involvement) computed by interpreting the same
// compound SSA function nodus_canvas_kpn.py actually emitted this canvas
// from -- see the harness's own comment block below for that derivation.
//
// This exercises the thread_manager.cpp fix that lets a headless-declared
// port on an unconnected boundary contact (module_io_rows Input row with no
// GP_CanvasEdgeDesc feeding it) actually reach the tick loop's consumption
// path, which previously silently saw val=0.0f regardless of what was pushed.
#include "nodus_headless_abi.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool push_f32(NodusHeadlessGraph* g, const char* name, int module_idx, int contact_idx, float value) {
    if (!nodus_headless_declare_input_port(g, name, module_idx, contact_idx, sizeof(float))) {
        std::fprintf(stderr, "declare_input_port(%s) failed\n", name);
        return false;
    }
    if (!nodus_headless_push_token(g, name, &value, sizeof(value))) {
        std::fprintf(stderr, "push_token(%s) failed\n", name);
        return false;
    }
    return true;
}

} // namespace

int main() {
    NodusHeadlessGraph* g = nodus_headless_create();
    if (!g) {
        std::fprintf(stderr, "nodus_headless_create failed\n");
        return 1;
    }

    if (!nodus_headless_load_file(g, "generated_program_canvas_collapsed.txt")) {
        std::fprintf(stderr, "load_file failed\n");
        nodus_headless_destroy(g);
        return 1;
    }

    // Verified module indices (nodus_canvas_kpn.py's collapse_canvas_regions
    // on the real 8-region pipeline, confirmed via direct inspection of the
    // generated document, not assumed):
    //   0 blend    in=2 out=1   boundary contacts [0,1]
    //   1 contrast in=2 out=1   boundary contacts [0,1]
    //   2 balance  in=2 out=1   boundary contacts [0,1]
    //   3 polish   in=8 out=1   boundary contacts [5,6] (curve's two raw
    //                           inputs, surviving after curve->envelope->
    //                           mixdown->polish inlined all the way through);
    //                           contacts 0,1,2,3,4,7 are fed by real edges
    //                           from blend/contrast/balance and need no push;
    //                           output contact = in_count(8) = 8.
    constexpr int kBlend = 0, kContrast = 1, kBalance = 2, kPolish = 3;

    const float blend_a = 0.7f, blend_b = -0.3f;
    const float contrast_a = 1.4f, contrast_b = 0.9f;
    const float balance_a = 0.5f, balance_b = 2.0f;
    const float curve_a = 1.2f, curve_b = 3.0f;

    bool ok = true;
    ok &= push_f32(g, "blend_a", kBlend, 0, blend_a);
    ok &= push_f32(g, "blend_b", kBlend, 1, blend_b);
    ok &= push_f32(g, "contrast_a", kContrast, 0, contrast_a);
    ok &= push_f32(g, "contrast_b", kContrast, 1, contrast_b);
    ok &= push_f32(g, "balance_a", kBalance, 0, balance_a);
    ok &= push_f32(g, "balance_b", kBalance, 1, balance_b);
    ok &= push_f32(g, "curve_a", kPolish, 5, curve_a);
    ok &= push_f32(g, "curve_b", kPolish, 6, curve_b);
    if (!ok) {
        nodus_headless_destroy(g);
        return 1;
    }

    if (!nodus_headless_declare_output_port(g, "result", kPolish, /*contact=*/8, sizeof(float))) {
        std::fprintf(stderr, "declare_output_port(result) failed\n");
        nodus_headless_destroy(g);
        return 1;
    }

    int32_t reached = nodus_headless_run_to_quiescence(g, 1.0 / 60.0, /*max_ticks=*/600);
    std::printf("run_to_quiescence: %s\n", reached ? "quiescent" : "max_ticks reached");

    float out_value = 0.0f;
    size_t written = 0;
    bool pulled = nodus_headless_pull_token(g, "result", &out_value, sizeof(out_value), &written) &&
                  written == sizeof(out_value);

    NodusHeadlessEvent events[32];
    int32_t n = nodus_headless_poll_events(g, events, 32);
    for (int32_t i = 0; i < n; ++i) {
        std::printf("event: kind=%d module=%d edge=%d tick=%llu\n",
            events[i].kind, events[i].module_idx, events[i].edge_idx,
            static_cast<unsigned long long>(events[i].tick_id));
    }

    if (!pulled) {
        std::printf("FAIL: no value pulled from 'result'\n");
        nodus_headless_destroy(g);
        return 1;
    }

    // Independent Python oracle (plain math, computed by interpreting the
    // same compound Function nodus_canvas_kpn.py's collapse emitted this
    // canvas from -- see the turing-side script that generated this file):
    //   blend_out    = tanh(blend_a + blend_b)
    //   contrast_out = sqrt(abs(contrast_a - contrast_b))
    //   balance_out  = sin(balance_a * balance_b)
    //   curve_out    = floor(curve_a ** curve_b)
    //   envelope_out = tanh(blend_out + contrast_out + balance_out + curve_out)
    //   gate_out     = float(blend_out > contrast_out)
    //   mixdown_out  = tanh(gate_out + envelope_out * balance_out)
    //   polish_out   = cos(sqrt(abs(mixdown_out)))
    // yielding 0.677036084 for these exact inputs.
    const double expected = 0.677036084;
    const double diff = std::fabs(static_cast<double>(out_value) - expected);
    std::printf("nodus computed = %.9f\n", static_cast<double>(out_value));
    std::printf("python expected = %.9f\n", expected);
    std::printf("abs diff = %.9g\n", diff);
    if (diff > 1e-4) {
        std::printf("FAIL: nodus output does not match the independent reference\n");
        nodus_headless_destroy(g);
        return 1;
    }
    std::printf("PASS: nodus's real execution matches the independent reference\n");

    // --- Throughput benchmark ---------------------------------------------
    // Same loaded graph, same 8 declared ports (already bound to their real
    // table edges above) -- reused, not recreated, so graph load / port
    // declaration / tool registration are excluded from the timed region.
    // Each sample: push 8 floats, run to quiescence on this small settled
    // graph, pull 1 float. Compared against a single fully vectorized NumPy
    // pass over the identical formula, run separately (see the paired
    // Python script) -- the fairest form of each side: nodus's real
    // per-sample dataflow-scheduler overhead vs NumPy's batched SIMD kernel.
    const char* bench_env = std::getenv("NODUS_BENCH_N");
    const int bench_n = bench_env ? std::atoi(bench_env) : 20000;
    if (bench_n > 0) {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> small(-2.0f, 2.0f);
        std::uniform_real_distribution<float> pow_safe(0.5f, 2.0f); // keep curve_a**curve_b finite/real

        // Pre-generate so RNG cost never lands inside the timed region.
        std::vector<float> ba(bench_n), bb(bench_n), ca(bench_n), cb(bench_n),
            va(bench_n), vb(bench_n), cua(bench_n), cub(bench_n);
        for (int i = 0; i < bench_n; ++i) {
            ba[i] = small(rng); bb[i] = small(rng);
            ca[i] = small(rng); cb[i] = small(rng);
            va[i] = small(rng); vb[i] = small(rng);
            cua[i] = pow_safe(rng); cub[i] = pow_safe(rng);
        }

        int completed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_n; ++i) {
            nodus_headless_push_token(g, "blend_a", &ba[i], sizeof(float));
            nodus_headless_push_token(g, "blend_b", &bb[i], sizeof(float));
            nodus_headless_push_token(g, "contrast_a", &ca[i], sizeof(float));
            nodus_headless_push_token(g, "contrast_b", &cb[i], sizeof(float));
            nodus_headless_push_token(g, "balance_a", &va[i], sizeof(float));
            nodus_headless_push_token(g, "balance_b", &vb[i], sizeof(float));
            nodus_headless_push_token(g, "curve_a", &cua[i], sizeof(float));
            nodus_headless_push_token(g, "curve_b", &cub[i], sizeof(float));
            nodus_headless_run_to_quiescence(g, 1.0 / 60.0, /*max_ticks=*/600);
            float discard = 0.0f;
            size_t got = 0;
            if (nodus_headless_pull_token(g, "result", &discard, sizeof(discard), &got) && got == sizeof(discard)) {
                ++completed;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        std::printf("\n--- nodus KPN throughput (graph reused, load cost excluded) ---\n");
        std::printf("samples: %d (completed: %d)\n", bench_n, completed);
        std::printf("elapsed: %.6f s\n", seconds);
        std::printf("throughput: %.1f samples/sec\n", bench_n / seconds);
        std::printf("per-sample: %.3f microseconds\n", (seconds * 1e6) / bench_n);
    }

    nodus_headless_destroy(g);
    return 0;
}
