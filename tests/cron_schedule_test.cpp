#include "cron_schedule.h"

#include <chrono>
#include <filesystem>
#include <cstdio>
#include <string>
#include <vector>

#define CHECK(x) do { if(!(x)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #x); return 1; } } while(0)

using namespace nodus::cron;

static bool leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

// Independent epoch builder: counts days with plain loops.
static int64_t utc(int y, int m, int d, int H, int M) {
    int64_t days = 0;
    for (int yy = 1970; yy < y; ++yy) days += leap(yy) ? 366 : 365;
    static const int ml[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    for (int mm = 1; mm < m; ++mm) days += ml[mm - 1] + ((mm == 2 && leap(y)) ? 1 : 0);
    days += d - 1;
    return days * 86400 + H * 3600 + M * 60;
}

static bool pe(const char* s, Expr& e) { std::string w; return parse_expr(s, e, &w); }

int main() {
    // sanity of the local helper itself
    CHECK(utc(1970, 1, 1, 0, 0) == 0);
    CHECK(utc(2000, 3, 1, 0, 0) == 951868800);

    // ---- parsing
    Expr e;
    CHECK(pe("* * * * *", e));
    CHECK(e.minutes == ((uint64_t(1) << 60) - 1) && e.day_any && e.weekday_any);
    CHECK(pe("*/15 9-17 * * 1-5", e));
    CHECK(e.minutes == ((1ull << 0) | (1ull << 15) | (1ull << 30) | (1ull << 45)));
    CHECK(e.hours == 0x3FE00u);  // bits 9..17
    CHECK(e.weekdays == 0x3Eu && !e.weekday_any && e.day_any);
    CHECK(pe("0 3 * * *", e));
    CHECK(e.minutes == 1 && e.hours == (1u << 3));
    CHECK(pe("5,35 0 1 1,7 *", e));
    CHECK(e.minutes == ((1ull << 5) | (1ull << 35)) && e.days == 2u && e.months == ((1u << 1) | (1u << 7)));
    CHECK(!e.day_any);
    Expr s7, s0;
    CHECK(pe("0 0 * * 7", s7) && pe("0 0 * * 0", s0));
    CHECK(s7.weekdays == s0.weekdays && s0.weekdays == 1);
    Expr junk;
    std::string why;
    CHECK(!parse_expr("* * * *", junk, &why) && !why.empty());
    CHECK(!pe("* * * * * *", junk));
    CHECK(!pe("60 * * * *", junk));
    CHECK(!pe("* 24 * * *", junk));
    CHECK(!pe("*/0 * * * *", junk));
    CHECK(!pe("5-1 * * * *", junk));
    CHECK(!pe("* * * * MON", junk));
    CHECK(!pe("", junk));
    CHECK(!pe("* * 0 * *", junk));
    CHECK(!pe("* * * 13 *", junk));
    CHECK(!pe("* * * * 8", junk));
    CHECK(!pe("5/10 * * * *", junk));
    CHECK(!pe("@daily", junk));

    // ---- matches
    CHECK(pe("0 3 * * *", e));
    CHECK(matches(e, utc(2026, 11, 11, 3, 0) + 59));
    CHECK(!matches(e, utc(2026, 11, 11, 3, 1)));

    // ---- next_after / last_at_or_before
    CHECK(pe("* * * * *", e));
    int64_t t0 = utc(2026, 11, 11, 10, 30);
    CHECK(next_after(e, t0) == t0 + 60);
    CHECK(next_after(e, t0 + 5) == t0 + 60);
    CHECK(last_at_or_before(e, t0) == t0);
    CHECK(last_at_or_before(e, t0 + 59) == t0);

    CHECK(pe("0 * * * *", e));
    CHECK(next_after(e, utc(2026, 11, 11, 10, 30)) == utc(2026, 11, 11, 11, 0));
    CHECK(next_after(e, utc(2026, 11, 11, 11, 0)) == utc(2026, 11, 11, 12, 0));
    CHECK(last_at_or_before(e, utc(2026, 11, 11, 11, 0)) == utc(2026, 11, 11, 11, 0));
    CHECK(last_at_or_before(e, utc(2026, 11, 11, 10, 59)) == utc(2026, 11, 11, 10, 0));

    CHECK(pe("0 3 * * *", e));
    CHECK(next_after(e, utc(2026, 11, 30, 3, 0)) == utc(2026, 12, 1, 3, 0));
    CHECK(next_after(e, utc(2026, 12, 31, 12, 0)) == utc(2027, 1, 1, 3, 0));
    CHECK(next_after(e, utc(2026, 12, 31, 2, 59)) == utc(2026, 12, 31, 3, 0));
    CHECK(last_at_or_before(e, utc(2027, 1, 1, 2, 0)) == utc(2026, 12, 31, 3, 0));
    CHECK(last_at_or_before(e, utc(2026, 12, 1, 2, 0)) == utc(2026, 11, 30, 3, 0));
    CHECK(last_at_or_before(e, utc(2026, 12, 1, 3, 0)) == utc(2026, 12, 1, 3, 0));

    CHECK(pe("0 0 29 2 *", e));
    CHECK(next_after(e, utc(2026, 1, 1, 0, 0)) == utc(2028, 2, 29, 0, 0));
    CHECK(last_at_or_before(e, utc(2027, 1, 1, 0, 0)) == utc(2024, 2, 29, 0, 0));
    CHECK(pe("0 0 31 2 *", e));
    CHECK(next_after(e, utc(2026, 1, 1, 0, 0)) == 0);
    CHECK(last_at_or_before(e, utc(2026, 1, 1, 0, 0)) == 0);

    // classic either-day rule: 13th OR Friday. 2026-11-13 is a Friday and the 13th.
    CHECK(pe("0 0 13 * 5", e));
    CHECK(matches(e, utc(2026, 11, 13, 0, 0)));
    CHECK(matches(e, utc(2026, 11, 6, 0, 0)));   // a Friday, not the 13th
    CHECK(matches(e, utc(2026, 12, 13, 0, 0)));  // 13th, a Sunday
    CHECK(!matches(e, utc(2026, 11, 12, 0, 0)));
    CHECK(next_after(e, utc(2026, 11, 7, 0, 0)) == utc(2026, 11, 13, 0, 0));
    CHECK(next_after(e, utc(2026, 11, 13, 0, 0)) == utc(2026, 11, 20, 0, 0));
    CHECK(next_after(e, utc(2026, 11, 27, 0, 0)) == utc(2026, 12, 4, 0, 0));
    CHECK(next_after(e, utc(2026, 12, 4, 0, 0)) == utc(2026, 12, 11, 0, 0));
    CHECK(next_after(e, utc(2026, 12, 11, 0, 0)) == utc(2026, 12, 13, 0, 0));
    // only dow restricted: dom must not matter
    CHECK(pe("0 0 * * 1", e));
    CHECK(next_after(e, utc(2026, 11, 11, 0, 0)) == utc(2026, 11, 16, 0, 0));  // Monday
    // only dom restricted
    CHECK(pe("0 0 15 * *", e));
    CHECK(next_after(e, utc(2026, 11, 11, 0, 0)) == utc(2026, 11, 15, 0, 0));

    // ---- parse_table
    std::vector<Task> tasks;
    const std::string table =
        "# comment\n"
        "\n"
        "  nightly 0 3 * * * missed=all max_late=7200 max_catch_up=5\r\n"
        "ping  */5   *  *  *  *\n"
        "   # indented comment\n"
        "weekly.job:1 30 4 * * 0 max_catch_up=1000 missed=skip\n";
    CHECK(parse_table(table, tasks, &why));
    CHECK(tasks.size() == 3);
    CHECK(tasks[0].name == "nightly" && tasks[0].missed == Missed::All && tasks[0].max_late_s == 7200 &&
          tasks[0].max_catch_up == 5 && tasks[0].expr_text == "0 3 * * *");
    CHECK(tasks[1].name == "ping" && tasks[1].missed == Missed::Once && tasks[1].max_late_s == 0 &&
          tasks[1].max_catch_up == 8 && tasks[1].expr_text == "*/5 * * * *");
    CHECK(tasks[2].name == "weekly.job:1" && tasks[2].missed == Missed::Skip && tasks[2].max_catch_up == 1000);

    std::vector<Task> bad;
    CHECK(!parse_table("a * * * * *\nb * * * * *\na * * * * *\n", bad, &why));
    CHECK(bad.empty() && why.find("3") != std::string::npos);
    CHECK(!parse_table("# c\na * * * * * color=red\n", bad, &why));
    CHECK(bad.empty() && why.find("2") != std::string::npos);
    CHECK(!parse_table("\n\nBad_Name * * * * *\n", bad, &why));
    CHECK(why.find("3") != std::string::npos);
    CHECK(!parse_table("ok * * * * *\nx 61 * * * *\n", bad, &why));
    CHECK(bad.empty() && why.find("line 2") != std::string::npos);
    CHECK(!parse_table("a * * * * * missed=once missed=all\n", bad, &why));
    CHECK(!parse_table("a * * * * * max_catch_up=0\n", bad, &why));
    CHECK(!parse_table("a * * * * * max_catch_up=1001\n", bad, &why));
    CHECK(!parse_table("a * * * * * missed=sometimes\n", bad, &why));
    CHECK(!parse_table("a * * * *\n", bad, &why));

    // ---- decide
    auto mk = [](const char* name, const char* ex, Missed m) {
        Task t;
        t.name = name;
        t.expr_text = ex;
        std::string w;
        parse_expr(ex, t.expr, &w);
        t.missed = m;
        return t;
    };
    {
        Task t = mk("nightly", "0 3 * * *", Missed::Once);
        int64_t last = utc(2026, 11, 10, 3, 0);
        // nothing due
        Decision d = decide(t, last, utc(2026, 11, 10, 12, 0));
        CHECK(d.fire.empty() && d.skipped == 0 && d.last_due == last && d.reason.empty());
        d = decide(t, last, last);
        CHECK(d.fire.empty() && d.last_due == last && d.reason.empty());

        // one on-time occurrence
        d = decide(t, last, utc(2026, 11, 11, 3, 0) + 20);
        CHECK(d.fire.size() == 1 && !d.fire[0].catch_up && d.fire[0].stands_for == 0 &&
              d.fire[0].scheduled_for == utc(2026, 11, 11, 3, 0) && d.skipped == 0);
        CHECK(d.last_due == utc(2026, 11, 11, 3, 0) && !d.reason.empty());

        // gap of 3 nightly runs (11,12,13 at 03:00); now 13th 10:00 -> all three missed
        int64_t now = utc(2026, 11, 13, 10, 0);
        d = decide(t, last, now);
        CHECK(d.fire.size() == 1 && d.fire[0].catch_up && d.fire[0].stands_for == 2 &&
              d.fire[0].scheduled_for == utc(2026, 11, 13, 3, 0) && d.skipped == 0);
        CHECK(d.last_due == utc(2026, 11, 13, 3, 0));
        CHECK(d.reason.find("3") != std::string::npos && d.reason.find("once") != std::string::npos);

        t.missed = Missed::All;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 3 && d.skipped == 0);
        CHECK(d.fire[0].scheduled_for == utc(2026, 11, 11, 3, 0) && d.fire[1].scheduled_for == utc(2026, 11, 12, 3, 0) &&
              d.fire[2].scheduled_for == utc(2026, 11, 13, 3, 0));
        CHECK(d.fire[0].catch_up && d.fire[1].catch_up && d.fire[2].catch_up);
        CHECK(d.last_due == utc(2026, 11, 13, 3, 0));

        t.missed = Missed::Skip;
        d = decide(t, last, now);
        CHECK(d.fire.empty() && d.skipped == 3 && d.last_due == utc(2026, 11, 13, 3, 0));
        CHECK(d.reason.find("skip") != std::string::npos && d.reason.find("3") != std::string::npos);

        // max_late drops old ones: only the 13th (7h old) is within 8h
        t.missed = Missed::All;
        t.max_late_s = 8 * 3600;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 1 && d.fire[0].scheduled_for == utc(2026, 11, 13, 3, 0) && d.skipped == 2);
        CHECK(d.last_due == utc(2026, 11, 13, 3, 0));
        t.missed = Missed::Once;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 1 && d.fire[0].stands_for == 0 && d.skipped == 2);
        t.max_late_s = 60;  // all dropped
        d = decide(t, last, now);
        CHECK(d.fire.empty() && d.skipped == 3 && d.last_due == utc(2026, 11, 13, 3, 0));
        t.max_late_s = 0;

        // max_catch_up caps all
        t.missed = Missed::All;
        t.max_catch_up = 2;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 2 && d.skipped == 1);
        CHECK(d.fire[0].scheduled_for == utc(2026, 11, 12, 3, 0) && d.fire[1].scheduled_for == utc(2026, 11, 13, 3, 0));

        // missed + on-time in same decision under once
        t.missed = Missed::Once;
        now = utc(2026, 11, 13, 3, 0) + 30;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 1 && !d.fire[0].catch_up && d.fire[0].stands_for == 2 &&
              d.fire[0].scheduled_for == utc(2026, 11, 13, 3, 0));
        CHECK(d.skipped == 0 && d.last_due == utc(2026, 11, 13, 3, 0));

        // all + on-time: catch-ups first then on-time
        t.missed = Missed::All;
        t.max_catch_up = 8;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 3 && d.fire[0].catch_up && d.fire[1].catch_up && !d.fire[2].catch_up);
    }
    {
        // every-minute task after a 30-day gap
        Task t = mk("ping", "* * * * *", Missed::Once);
        int64_t now = utc(2026, 11, 11, 12, 0);
        int64_t last = now - 30 * 86400;
        auto c0 = std::chrono::steady_clock::now();
        Decision d = decide(t, last, now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - c0).count();
        CHECK(ms < 5000);
        // minutes now-60 and now are on time (<=90s); everything older is missed
        CHECK(d.fire.size() == 2 && !d.fire[0].catch_up && !d.fire[1].catch_up);
        CHECK(d.fire[0].scheduled_for == now - 60 && d.fire[1].scheduled_for == now);
        CHECK(d.fire[0].stands_for == 30u * 1440u - 2u && d.fire[1].stands_for == 0);
        CHECK(d.last_due == now);
        CHECK(d.skipped == 0);

        t.missed = Missed::All;
        t.max_catch_up = 10;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 12);
        CHECK(d.skipped == (30u * 1440u - 2u) - 10u);
        CHECK(d.last_due == now);

        // beyond the walk cap: 200 days
        t.max_catch_up = 3;
        last = now - 200 * 86400;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 5 && d.last_due == now && d.skipped > 100000);
        CHECK(d.fire[0].scheduled_for == now - 4 * 60);
        CHECK(d.fire[4].scheduled_for == now && !d.fire[4].catch_up);
        CHECK(d.reason.find("at least") != std::string::npos);
        // just beyond the cap (80 days = 115200 min): exact join
        last = now - 80 * 86400;
        d = decide(t, last, now);
        CHECK(d.fire.size() == 5 && d.skipped == (80u * 1440u - 2u) - 3u);
        CHECK(d.reason.find("at least") == std::string::npos);
    }

    // ---- ledger
    {
        const std::string path = "cron_schedule_test_ledger.tmp.txt";
        std::remove(path.c_str());
        Ledger l;
        CHECK(load_ledger(path, l, &why));
        CHECK(l.alive == 0 && l.tasks.empty());

        Ledger a;
        a.alive = 1789000000;
        TaskState ts;
        ts.last_due = 1789000020;
        ts.last_started = 1789000030;
        ts.last_finished = 1789000040;
        ts.last_status = 18446744073709551615ull;
        ts.fired = 12;
        ts.skipped = 3;
        a.tasks["nightly"] = ts;
        TaskState t2;
        t2.last_due = 7;
        a.tasks["x.y:z-1"] = t2;
        CHECK(save_ledger(path, a, &why));
        std::vector<char> dummy;
        Ledger b;
        CHECK(load_ledger(path, b, &why));
        CHECK(b.alive == a.alive && b.tasks.size() == 2);
        const TaskState& r = b.tasks["nightly"];
        CHECK(r.last_due == ts.last_due && r.last_started == ts.last_started && r.last_finished == ts.last_finished &&
              r.last_status == ts.last_status && r.fired == 12 && r.skipped == 3);
        CHECK(b.tasks["x.y:z-1"].last_due == 7);
        // overwrite works and temp file is gone
        a.alive = 5;
        CHECK(save_ledger(path, a, &why));
        CHECK(load_ledger(path, b, &why) && b.alive == 5);
        CHECK(!std::filesystem::exists(path + ".tmp"));
        std::remove(path.c_str());

        // corrupt first line
        {
            std::FILE* f = std::fopen(path.c_str(), "wb");
            CHECK(f != nullptr);
            std::fputs("cron-ledger 2\nalive 5\n", f);
            std::fclose(f);
        }
        Ledger c;
        c.alive = 99;
        CHECK(!load_ledger(path, c, &why) && !why.empty());
        CHECK(c.alive == 0 && c.tasks.empty());
        // malformed task line
        {
            std::FILE* f = std::fopen(path.c_str(), "wb");
            CHECK(f != nullptr);
            std::fputs("cron-ledger 1\nalive 5\ntask n 1 2 3\n", f);
            std::fclose(f);
        }
        CHECK(!load_ledger(path, c, &why) && c.tasks.empty() && c.alive == 0);
        std::remove(path.c_str());
    }

    std::printf("OK cron_schedule_test all checks passed\n");
    return 0;
}
