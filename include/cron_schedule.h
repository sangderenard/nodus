// cron_schedule.h - pure cron schedule core (UTC, epoch seconds in, no clock, no threads).
// Header-only C++17, standard library only.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace nodus {
namespace cron {

struct Expr {
    uint64_t minutes = 0;
    uint32_t hours = 0;
    uint32_t days = 0;
    uint16_t months = 0;
    uint8_t weekdays = 0;
    bool day_any = true;
    bool weekday_any = true;
};

// ---------------------------------------------------------------- civil time
namespace detail {

inline int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// Howard Hinnant: days since 1970-01-01 for a proleptic Gregorian date.
inline int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

inline void civil_from_days(int64_t z, int64_t& y, unsigned& m, unsigned& d) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = static_cast<int64_t>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp < 10 ? mp + 3 : mp - 9;
    y += (m <= 2);
}

struct Civil {
    int64_t year;
    unsigned month, day;
    int hour, minute;
    int weekday;      // 0 = Sunday
    int64_t day_num;  // days since epoch
};

inline Civil split(int64_t t) {
    Civil c;
    const int64_t dn = floor_div(t, 86400);
    const int64_t sod = t - dn * 86400;
    civil_from_days(dn, c.year, c.month, c.day);
    c.hour = static_cast<int>(sod / 3600);
    c.minute = static_cast<int>((sod % 3600) / 60);
    c.day_num = dn;
    int64_t w = (dn + 4) % 7;  // 1970-01-01 was Thursday (4)
    if (w < 0) w += 7;
    c.weekday = static_cast<int>(w);
    return c;
}

inline bool day_ok(const Expr& e, const Civil& c) {
    const bool dom = (e.days >> c.day) & 1u;
    const bool dow = (e.weekdays >> c.weekday) & 1u;
    if (!e.day_any && !e.weekday_any) return dom || dow;
    if (e.day_any && e.weekday_any) return true;
    if (e.day_any) return dow;
    return dom;
}

inline std::string fmt_time(int64_t t) {
    const Civil c = split(t);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04lld-%02u-%02u %02d:%02d UTC",
                  static_cast<long long>(c.year), c.month, c.day, c.hour, c.minute);
    return buf;
}

inline bool set_why(std::string* why, const std::string& s) {
    if (why) *why = s;
    return false;
}

inline std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '\f' || ch == '\v') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

inline bool parse_uint(const std::string& s, uint64_t& v, uint64_t max_value) {
    if (s.empty() || s.size() > 18) return false;
    uint64_t r = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') return false;
        r = r * 10 + static_cast<uint64_t>(ch - '0');
    }
    if (r > max_value) return false;
    v = r;
    return true;
}

// Parses one field into a bit mask over [lo, hi]. Sets is_star when the text is exactly "*".
inline bool parse_field(const std::string& text, unsigned lo, unsigned hi, uint64_t& mask,
                        bool& is_star, std::string& why) {
    mask = 0;
    is_star = (text == "*");
    if (text.empty()) { why = "empty field"; return false; }
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t comma = text.find(',', pos);
        if (comma == std::string::npos) comma = text.size();
        const std::string item = text.substr(pos, comma - pos);
        pos = comma + 1;
        if (item.empty()) { why = "empty list item in '" + text + "'"; return false; }
        std::string base = item;
        uint64_t step = 1;
        bool has_step = false;
        const size_t slash = item.find('/');
        if (slash != std::string::npos) {
            base = item.substr(0, slash);
            if (!parse_uint(item.substr(slash + 1), step, 1000000) || step < 1) {
                why = "bad step in '" + item + "'";
                return false;
            }
            has_step = true;
        }
        uint64_t a = lo, b = hi;
        bool is_range = false;
        if (base == "*") {
            // full range
        } else {
            const size_t dash = base.find('-');
            if (dash == std::string::npos) {
                if (has_step) { why = "step needs '*' or a range in '" + item + "'"; return false; }
                if (!parse_uint(base, a, 1000000)) { why = "bad value '" + base + "'"; return false; }
                b = a;
            } else {
                is_range = true;
                if (!parse_uint(base.substr(0, dash), a, 1000000) ||
                    !parse_uint(base.substr(dash + 1), b, 1000000)) {
                    why = "bad range '" + base + "'";
                    return false;
                }
            }
        }
        (void)is_range;
        if (a < lo || a > hi || b < lo || b > hi) {
            why = "value out of range in '" + item + "'";
            return false;
        }
        if (a > b) { why = "range start is after its end in '" + item + "'"; return false; }
        for (uint64_t v = a; v <= b; v += step) mask |= (uint64_t(1) << v);
    }
    if (mask == 0) { why = "field selects nothing"; return false; }
    return true;
}

}  // namespace detail

inline bool parse_expr(const std::string& text, Expr& out, std::string* why = nullptr) {
    const std::vector<std::string> f = detail::split_ws(text);
    if (f.size() != 5) {
        return detail::set_why(why, "expected exactly 5 fields, found " + std::to_string(f.size()));
    }
    static const char* names[5] = {"minute", "hour", "day-of-month", "month", "day-of-week"};
    static const unsigned lo[5] = {0, 0, 1, 1, 0};
    static const unsigned hi[5] = {59, 23, 31, 12, 7};
    uint64_t mask[5] = {0, 0, 0, 0, 0};
    bool star[5] = {false, false, false, false, false};
    for (int i = 0; i < 5; ++i) {
        std::string w;
        if (!detail::parse_field(f[i], lo[i], hi[i], mask[i], star[i], w)) {
            return detail::set_why(why, std::string(names[i]) + " field: " + w);
        }
    }
    if (mask[4] & (uint64_t(1) << 7)) mask[4] = (mask[4] & ~(uint64_t(1) << 7)) | 1u;
    Expr e;
    e.minutes = mask[0];
    e.hours = static_cast<uint32_t>(mask[1]);
    e.days = static_cast<uint32_t>(mask[2]);
    e.months = static_cast<uint16_t>(mask[3]);
    e.weekdays = static_cast<uint8_t>(mask[4]);
    e.day_any = star[2];
    e.weekday_any = star[4];
    out = e;
    return true;
}

inline bool matches(const Expr& e, int64_t epoch_seconds) {
    const detail::Civil c = detail::split(epoch_seconds);
    if (!((e.months >> c.month) & 1u)) return false;
    if (!((e.hours >> c.hour) & 1u)) return false;
    if (!((e.minutes >> c.minute) & 1u)) return false;
    return detail::day_ok(e, c);
}

namespace detail {
constexpr int64_t kSearchSpan = 5 * 366 * 86400;
}

inline int64_t next_after(const Expr& e, int64_t after) {
    const int64_t limit = after + detail::kSearchSpan;
    int64_t t = detail::floor_div(after, 60) * 60 + 60;
    while (t <= limit) {
        const detail::Civil c = detail::split(t);
        if (!((e.months >> c.month) & 1u)) {
            int64_t y = c.year;
            unsigned m = c.month + 1;
            if (m > 12) { m = 1; ++y; }
            t = detail::days_from_civil(y, m, 1) * 86400;
            continue;
        }
        if (!detail::day_ok(e, c)) {
            t = (c.day_num + 1) * 86400;
            continue;
        }
        if (!((e.hours >> c.hour) & 1u)) {
            t = c.day_num * 86400 + static_cast<int64_t>(c.hour + 1) * 3600;
            continue;
        }
        if (!((e.minutes >> c.minute) & 1u)) {
            t += 60;
            continue;
        }
        return t;
    }
    return 0;
}

inline int64_t last_at_or_before(const Expr& e, int64_t at) {
    const int64_t limit = at - detail::kSearchSpan;
    int64_t t = detail::floor_div(at, 60) * 60;
    while (t >= limit) {
        const detail::Civil c = detail::split(t);
        if (!((e.months >> c.month) & 1u)) {
            t = detail::days_from_civil(c.year, c.month, 1) * 86400 - 60;
            continue;
        }
        if (!detail::day_ok(e, c)) {
            t = c.day_num * 86400 - 60;
            continue;
        }
        if (!((e.hours >> c.hour) & 1u)) {
            t = c.day_num * 86400 + static_cast<int64_t>(c.hour) * 3600 - 60;
            continue;
        }
        if (!((e.minutes >> c.minute) & 1u)) {
            t -= 60;
            continue;
        }
        return t;
    }
    return 0;
}

// ---------------------------------------------------------------- table
enum class Missed { Skip, Once, All };

struct Task {
    std::string name;
    std::string expr_text;
    Expr expr;
    Missed missed = Missed::Once;
    int64_t max_late_s = 0;
    uint32_t max_catch_up = 8;
};

namespace detail {
inline bool valid_name(const std::string& n) {
    if (n.empty() || n.size() > 64) return false;
    auto alnum = [](char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); };
    if (!alnum(n[0])) return false;
    for (size_t i = 1; i < n.size(); ++i) {
        const char c = n[i];
        if (!(alnum(c) || c == '.' || c == '_' || c == ':' || c == '-')) return false;
    }
    return true;
}
}  // namespace detail

inline bool parse_table(const std::string& text, std::vector<Task>& out, std::string* why = nullptr) {
    out.clear();
    std::vector<Task> tasks;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    auto fail = [&](const std::string& msg) {
        return detail::set_why(why, "line " + std::to_string(lineno) + ": " + msg);
    };
    while (std::getline(in, line)) {
        ++lineno;
        const std::vector<std::string> tok = detail::split_ws(line);
        if (tok.empty() || tok[0][0] == '#') continue;
        if (tok.size() < 6) return fail("expected '<name> <5 cron fields> [options]'");
        Task t;
        t.name = tok[0];
        if (!detail::valid_name(t.name)) return fail("bad task name '" + t.name + "'");
        for (const Task& o : tasks) {
            if (o.name == t.name) return fail("duplicate task name '" + t.name + "'");
        }
        t.expr_text = tok[1] + " " + tok[2] + " " + tok[3] + " " + tok[4] + " " + tok[5];
        std::string w;
        if (!parse_expr(t.expr_text, t.expr, &w)) return fail("bad expression: " + w);
        bool seen_missed = false, seen_late = false, seen_catch = false;
        for (size_t i = 6; i < tok.size(); ++i) {
            const std::string& o = tok[i];
            const size_t eq = o.find('=');
            if (eq == std::string::npos) return fail("unknown option '" + o + "'");
            const std::string key = o.substr(0, eq), val = o.substr(eq + 1);
            if (key == "missed") {
                if (seen_missed) return fail("duplicate option 'missed'");
                seen_missed = true;
                if (val == "skip") t.missed = Missed::Skip;
                else if (val == "once") t.missed = Missed::Once;
                else if (val == "all") t.missed = Missed::All;
                else return fail("missed must be skip, once or all");
            } else if (key == "max_late") {
                if (seen_late) return fail("duplicate option 'max_late'");
                seen_late = true;
                uint64_t v = 0;
                if (!detail::parse_uint(val, v, 4000000000000ULL)) return fail("bad max_late value '" + val + "'");
                t.max_late_s = static_cast<int64_t>(v);
            } else if (key == "max_catch_up") {
                if (seen_catch) return fail("duplicate option 'max_catch_up'");
                seen_catch = true;
                uint64_t v = 0;
                if (!detail::parse_uint(val, v, 1000) || v < 1) return fail("max_catch_up must be 1..1000");
                t.max_catch_up = static_cast<uint32_t>(v);
            } else {
                return fail("unknown option '" + key + "'");
            }
        }
        tasks.push_back(t);
    }
    out = tasks;
    return true;
}

// ---------------------------------------------------------------- decide
struct Firing {
    int64_t scheduled_for = 0;
    bool catch_up = false;
    uint32_t stands_for = 0;
};

struct Decision {
    std::vector<Firing> fire;
    uint32_t skipped = 0;
    int64_t last_due = 0;
    std::string reason;
};

inline Decision decide(const Task& task, int64_t last_due, int64_t now, int64_t on_time_s = 90) {
    Decision d;
    d.last_due = last_due;
    if (now <= last_due) return d;

    constexpr size_t kWalkCap = 100000;
    std::vector<int64_t> occ;
    bool exact = true;
    uint64_t unseen_older = 0;  // lower bound on occurrences not in `occ`
    {
        int64_t t = last_due;
        bool overflow = false;
        for (;;) {
            const int64_t n = next_after(task.expr, t);
            if (n == 0 || n > now) break;
            if (occ.size() >= kWalkCap) { overflow = true; break; }
            occ.push_back(n);
            t = n;
        }
        if (overflow) {
            const int64_t fwd_last = occ.back();
            std::vector<int64_t> back;  // newest first
            bool joined = false;
            int64_t at = now;
            while (back.size() < kWalkCap) {
                const int64_t p = last_at_or_before(task.expr, at);
                if (p == 0 || p <= last_due) { joined = true; break; }
                if (p <= fwd_last) { joined = true; break; }
                back.push_back(p);
                at = p - 60;
            }
            if (joined) {
                for (size_t i = back.size(); i-- > 0;) occ.push_back(back[i]);
            } else {
                exact = false;
                unseen_older = kWalkCap;
                occ.assign(back.rbegin(), back.rend());
            }
        }
    }
    if (occ.empty()) return d;
    d.last_due = occ.back();

    std::vector<int64_t> missed, ontime;
    uint64_t missed_total = unseen_older;
    uint64_t skipped = unseen_older;
    for (int64_t t : occ) {
        if (now - t <= on_time_s) { ontime.push_back(t); continue; }
        ++missed_total;
        if (task.max_late_s > 0 && now - t > task.max_late_s) { ++skipped; continue; }
        missed.push_back(t);
    }

    const char* policy = task.missed == Missed::Skip ? "skip" : task.missed == Missed::Once ? "once" : "all";
    std::vector<Firing> catchups;
    uint32_t ontime_extra = 0;
    int64_t once_target = 0;
    uint64_t once_extra = 0;
    uint64_t ran_missed = 0;
    switch (task.missed) {
        case Missed::Skip:
            skipped += missed.size();
            break;
        case Missed::Once:
            if (!missed.empty()) {
                if (!ontime.empty()) {
                    ontime_extra = static_cast<uint32_t>(missed.size());
                } else {
                    Firing f;
                    f.scheduled_for = missed.back();
                    f.catch_up = true;
                    f.stands_for = static_cast<uint32_t>(missed.size() - 1);
                    catchups.push_back(f);
                    once_target = f.scheduled_for;
                    once_extra = missed.size() - 1;
                }
                ran_missed = missed.size();
            }
            break;
        case Missed::All: {
            const size_t keep = task.max_catch_up;
            const size_t start = missed.size() > keep ? missed.size() - keep : 0;
            skipped += start;
            for (size_t i = start; i < missed.size(); ++i) {
                Firing f;
                f.scheduled_for = missed[i];
                f.catch_up = true;
                catchups.push_back(f);
            }
            ran_missed = missed.size() - start;
            break;
        }
    }

    d.fire = catchups;
    for (size_t i = 0; i < ontime.size(); ++i) {
        Firing f;
        f.scheduled_for = ontime[i];
        f.catch_up = false;
        f.stands_for = (i == 0) ? ontime_extra : 0;
        d.fire.push_back(f);
    }
    d.skipped = static_cast<uint32_t>(skipped > 0xFFFFFFFFULL ? 0xFFFFFFFFULL : skipped);

    // reason
    std::string r;
    const std::string atleast = exact ? "" : "at least ";
    if (missed_total == 0) {
        r = std::to_string(ontime.size()) + (ontime.size() == 1 ? " run of " : " runs of ") + task.name +
            " is due now and on time; running it.";
    } else {
        r = atleast + std::to_string(missed_total) + (missed_total == 1 ? " run of " : " runs of ") + task.name +
            (missed_total == 1 ? " was" : " were") + " missed during a gap; policy is " + policy + ", ";
        if (task.missed == Missed::Skip) {
            r += "so none of the missed runs will run.";
        } else if (task.missed == Missed::Once) {
            if (once_target != 0) {
                r += "running once now for the latest (" + detail::fmt_time(once_target) + "), which stands for the other " +
                     std::to_string(once_extra) + ".";
            } else if (ran_missed > 0) {
                r += "so the on-time run covers the " + std::to_string(ran_missed) + " missed.";
            } else {
                r += "but all of them are older than max_late, so none will run.";
            }
        } else {
            r += "running the most recent " + std::to_string(ran_missed) + " (up to max_catch_up " +
                 std::to_string(task.max_catch_up) + ")";
            r += ".";
        }
        if (task.max_late_s > 0) r += " Runs more than " + std::to_string(task.max_late_s) + "s late are dropped.";
        if (d.skipped > 0) r += " " + std::string(exact ? "" : "At least ") + std::to_string(d.skipped) + " will never be run.";
        if (!ontime.empty() && task.missed != Missed::Once) {
            r += " " + std::to_string(ontime.size()) + " on-time run" + (ontime.size() == 1 ? "" : "s") + " also fire.";
        }
    }
    d.reason = r;
    return d;
}

// ---------------------------------------------------------------- ledger
struct TaskState {
    int64_t last_due = 0;
    int64_t last_started = 0, last_finished = 0;
    uint64_t last_status = 0;
    uint64_t fired = 0, skipped = 0;
};

struct Ledger {
    int64_t alive = 0;
    std::map<std::string, TaskState> tasks;
};

inline bool load_ledger(const std::string& path, Ledger& out, std::string* why = nullptr) {
    out = Ledger();
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::path(path), ec)) return true;
    std::ifstream f(path, std::ios::binary);
    if (!f) return detail::set_why(why, "cannot open " + path);
    Ledger l;
    std::string line;
    int lineno = 0;
    auto bad = [&](const std::string& m) {
        out = Ledger();
        return detail::set_why(why, "ledger line " + std::to_string(lineno) + ": " + m);
    };
    while (std::getline(f, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (lineno == 1) {
            if (line != "cron-ledger 1") return bad("not a cron-ledger 1 file");
            continue;
        }
        if (line.empty()) continue;
        std::istringstream is(line);
        std::string kind;
        is >> kind;
        if (kind == "alive") {
            long long v;
            std::string extra;
            if (!(is >> v) || (is >> extra)) return bad("malformed alive record");
            l.alive = v;
        } else if (kind == "task") {
            std::string name, extra;
            long long a, b, c;
            unsigned long long s, fi, sk;
            if (!(is >> name >> a >> b >> c >> s >> fi >> sk) || (is >> extra)) return bad("malformed task record");
            if (l.tasks.count(name)) return bad("duplicate task '" + name + "'");
            TaskState ts;
            ts.last_due = a;
            ts.last_started = b;
            ts.last_finished = c;
            ts.last_status = s;
            ts.fired = fi;
            ts.skipped = sk;
            l.tasks[name] = ts;
        } else {
            return bad("unknown record '" + kind + "'");
        }
    }
    if (lineno < 1) return bad("empty file");
    out = l;
    return true;
}

inline bool save_ledger(const std::string& path, const Ledger& ledger, std::string* why = nullptr) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return detail::set_why(why, "cannot write " + tmp);
        f << "cron-ledger 1\n";
        f << "alive " << ledger.alive << "\n";
        for (const auto& kv : ledger.tasks) {
            const TaskState& t = kv.second;
            f << "task " << kv.first << ' ' << t.last_due << ' ' << t.last_started << ' ' << t.last_finished << ' '
              << t.last_status << ' ' << t.fired << ' ' << t.skipped << "\n";
        }
        f.flush();
        if (!f) return detail::set_why(why, "write failed for " + tmp);
    }
    std::error_code ec;
    std::filesystem::rename(std::filesystem::path(tmp), std::filesystem::path(path), ec);
    if (ec) return detail::set_why(why, "rename failed: " + ec.message());
    return true;
}

}  // namespace cron
}  // namespace nodus
