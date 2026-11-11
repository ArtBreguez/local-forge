"""Anti-debug techniques: name the technique and one detail read off the code.

Techniques: ptrace(PTRACE_TRACEME) through libc, the same through a raw syscall(SYS_ptrace, ...), a getenv() probe, a
clock_gettime / gettimeofday timing check, a TracerPid probe of /proc/self/status. The question lists the four technique words and
asks for one detail: the value returned on detection (ptrace, tracerpid), the environment variable (env), the threshold in
nanoseconds or microseconds (timing). The harness does not touch the real process: a preamble swaps the libc call for a fake
(`#define ptrace fake_ptrace`, a fake clock that moves by exactly D, a fake fopen on an in-memory status file), so the
test proves the request code, the syscall number, the path, the threshold and its unit, not just that the code compiles.

Every variant draws its constants from a wide range (return values 1..250, thresholds, work sizes, environment names from ~150
spellings), writes the test in several equivalent forms and starts with 1..3 decoy statements (core.decoys), so two tasks are
the same program only by accident (tools/rea-bench `check` reports duplicates).
"""
from __future__ import annotations

from .core import HDR, Task, dec, pick_int, with_decoys

FAMILY = "antidebug"
HEADERS = (HDR.rstrip("\n") + "\n#include <unistd.h>\n#include <time.h>\n#include <sys/time.h>\n#include <sys/ptrace.h>\n"
           "#include <sys/syscall.h>\n\n")

TECH = {"ptrace": ["ptrace", "ptrace_traceme", "traceme"], "timing": ["timing", "time-based", "clock_gettime", "gettimeofday", "rdtsc"],
        "env": ["env", "environment", "getenv"], "tracerpid": ["tracerpid"]}
OPTIONS = "one of: ptrace, timing, env, tracerpid"

# Environment variable names: none is an English word on its own (a reply that merely contains "lines" must not pass), all have an
# underscore or are long. About 25 real ones plus PREFIX_WORD[_N] combinations.
REAL_ENV = ["LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_LIBRARY_PATH", "LD_BIND_NOW", "LD_BIND_NOT", "LD_PROFILE", "LD_SHOW_AUXV",
            "LD_ASSUME_KERNEL", "LD_DEBUG_OUTPUT", "LD_ORIGIN_PATH", "GLIBC_TUNABLES", "MALLOC_PERTURB_", "MALLOC_CHECK_", "MALLOC_TRACE",
            "MALLOC_ARENA_MAX", "PYTHONINSPECT", "PYTHONDEBUG", "PYTHONFAULTHANDLER", "JAVA_TOOL_OPTIONS", "NODE_OPTIONS", "ASAN_OPTIONS",
            "UBSAN_OPTIONS", "VALGRIND_OPTS", "STRACE_LOG"]
ENV_PREFIX = ["APP", "SVC", "SYS", "RT", "DBG", "CFG", "SEC", "NET", "GUARD", "AGENT"]
ENV_WORD = ["TRACE", "ATTACH", "HOOK", "PROBE", "WATCH", "INSPECT", "DEBUGGER", "DUMP", "VERBOSE", "PROFILE", "TAP", "SPY", "LOGLEVEL", "SANDBOX"]
ENV_VARS = REAL_ENV + [f"{p}_{w}" for p in ENV_PREFIX for w in ENV_WORD]


def env_name(rng) -> str:
    name = rng.choice(REAL_ENV) if rng.random() < 0.3 else f"{rng.choice(ENV_PREFIX)}_{rng.choice(ENV_WORD)}"
    return name + (f"_{rng.randint(1, 99)}" if rng.random() < 0.3 and not name.endswith("_") else "")


TIMING_API = {"ns": "clock_gettime", "us": "gettimeofday"}     # the clock call each timing variant makes (round 8, m3)


def truth(tech, detail_alts, unit=None):
    """`unit` (timing only): the variant's own clock call. The name group of a timing task holds the generic words and THAT call: `rdtsc`, or the call of the
    other variant, names a technique the code does not use, so it is a wrong answer (it is in `reject`), not a spelling of "timing" (round 8, m3)."""
    names = TECH[tech]
    if tech == "timing":
        names = [n for n in names if n in ("timing", "time-based") or n == TIMING_API[unit]]
    other = [n for t, ns in TECH.items() if t != tech for n in ns] + [n for n in TECH[tech] if n not in names]
    return {"score": "tokens", "accept": [names, detail_alts], "reject": other}


def question(detail):
    return f"Which anti-debugging technique does proc implement ({OPTIONS}), and {detail}"


def returns(rng) -> int:
    """A return code 1..250; the held-out seed draws the multiples of 5, training seeds the other values (a ptrace check is a few lines,
    so without a reserved part a held-out task would have a training twin that differs only in its decoys)."""
    return pick_int(rng, 1, 250, 5)


RV_AT = r"(?:return |\? )([1-9]\d*)(?:;| : 0;)"        # where proc returns the asked code (never `return 0;`)


def _decoys(rng, body, forbidden):
    return with_decoys(body, rng, rng.choice([1, 1, 2, 2, 3]), forbidden)


def _ptrace(rng, raw):
    rv = returns(rng)
    form = rng.randrange(4)
    if raw:
        call = "syscall(SYS_ptrace, PTRACE_TRACEME, 0, 0, 0)"
        body = ["int proc(void) {\n    long r = %s;\n    return r != 0 ? %d : 0;\n}",
                "int proc(void) {\n    long r = %s;\n    if (r < 0) return %d;\n    return 0;\n}",
                "int proc(void) {\n    if (%s == -1) {\n        return %d;\n    }\n    return 0;\n}",
                "int proc(void) {\n    long r = %s;\n    if (r) return %d;\n    return 0;\n}"][form] % (call, rv)
        pre = """static long fake_rv, last_no, last_req;
static long fake_syscall(long no, ...) {
    va_list ap;
    va_start(ap, no);
    last_no = no;
    last_req = va_arg(ap, int);
    va_end(ap);
    return fake_rv;
}
#define syscall fake_syscall
"""
        proof = "    printf(\"%d\\n\", last_no == SYS_ptrace && last_req == PTRACE_TRACEME);\n"
    else:
        call = "ptrace(PTRACE_TRACEME, 0, NULL, NULL)"
        body = ["int proc(void) {\n    long r = %s;\n    return r == -1 ? %d : 0;\n}",
                "int proc(void) {\n    if (%s < 0) {\n        return %d;\n    }\n    return 0;\n}",
                "int proc(void) {\n    if (%s == -1) return %d;\n    return 0;\n}",
                "int proc(void) {\n    long r = %s;\n    if (r != 0) return %d;\n    return 0;\n}"][form] % (call, rv)
        pre = """static long fake_rv, last_req;
static long fake_ptrace(int req, ...) {
    last_req = req;
    return fake_rv;
}
#define ptrace fake_ptrace
"""
        proof = "    printf(\"%d\\n\", last_req == PTRACE_TRACEME);\n"
    body = _decoys(rng, body, {rv})
    main = (f'int main(void) {{\n    fake_rv = 0;\n    printf("%d\\n", proc());\n    fake_rv = -1;\n    printf("%d\\n", proc());\n{proof}    return 0;\n}}')
    pre = "#include <stdarg.h>\n" + pre
    return Task(kind=FAMILY, variant="ptrace_raw" if raw else "ptrace", headers=HEADERS, body=body,
                question=question("what does proc return when the call shows that a debugger is attached (decimal)?"),
                truth=truth("ptrace", dec(rv)), params={"rv": rv, "raw": raw, "form": form}, wrong=f"timing, {rv}", check=(main, f"0\n{rv}\n1\n", pre),
                facts=[[rv, {"at": RV_AT, "kind": "exact"}]])


def v_ptrace(rng):
    return _ptrace(rng, False)


def v_ptrace_raw(rng):
    return _ptrace(rng, True)


def v_env(rng):
    name, rv = env_name(rng), returns(rng)
    form = rng.randrange(4)
    body = [f'int proc(void) {{\n    if (getenv("{name}") != NULL) return {rv};\n    return 0;\n}}',
            f'int proc(void) {{\n    const char *e = getenv("{name}");\n    if (e && *e) return {rv};\n    return 0;\n}}',
            f'int proc(void) {{\n    return getenv("{name}") ? {rv} : 0;\n}}',
            f'int proc(void) {{\n    char *v = getenv("{name}");\n    if (v == NULL) {{\n        return 0;\n    }}\n    return {rv};\n}}'][form]
    body = _decoys(rng, body, {rv})
    main = (f'int main(void) {{\n    unsetenv("{name}");\n    printf("%d\\n", proc());\n    setenv("{name}", "1", 1);\n'
            f'    printf("%d\\n", proc());\n    return 0;\n}}')
    other = rng.choice([n for n in ENV_VARS if n != name])
    t = truth("env", [name.lower()])
    t["cand"] = "[a-z0-9]*_[a-z0-9_]*"       # another variable name in the headline ("LD_PRELOAD or LD_AUDIT") is a second candidate
    return Task(kind=FAMILY, variant="env", headers=HEADERS, body=body, question=question("which environment variable does it look at?"),
                truth=t, params={"name": name, "rv": rv, "form": form}, wrong=f"env, {other}", check=(main, f"0\n{rv}\n"), facts=[[name]])


def _timing(rng, unit):
    ns = unit == "ns"
    if rng.random() < 0.7:                     # a round number: k * 10^e
        thr = rng.randint(1, 999) * rng.choice([10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7] if ns else [10, 10 ** 2, 10 ** 3, 10 ** 4])
    else:
        thr = rng.randint(1000, 99_999_999) if ns else rng.randint(100, 999_999)
    work = rng.randint(64, 20000)
    form = rng.randrange(3)
    ret = ["return d > {t}LL;", "if (d > {t}LL) return 1;\n    return 0;", "return d > {t}LL ? 1 : 0;"][form].format(t=thr)
    if ns:
        body = f"""int proc(void) {{
    struct timespec a, b;
    volatile unsigned x = 0;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i = 0; i < {work}; i++) x += (unsigned)i;
    clock_gettime(CLOCK_MONOTONIC, &b);
    long long d = (long long)(b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec);
    {ret}
}}"""
        pre = """static long long fake_d;
static int fake_calls;
static int fake_clock_gettime(clockid_t id, struct timespec *t) {
    long long v = (fake_calls++ & 1) ? fake_d : 0;
    (void)id;
    t->tv_sec = v / 1000000000LL;
    t->tv_nsec = (long)(v % 1000000000LL);
    return 0;
}
#define clock_gettime fake_clock_gettime
"""
        name = "nanoseconds"
    else:
        body = f"""int proc(void) {{
    struct timeval a, b;
    volatile unsigned x = 0;
    gettimeofday(&a, NULL);
    for (int i = 0; i < {work}; i++) x += (unsigned)i;
    gettimeofday(&b, NULL);
    long long d = (long long)(b.tv_sec - a.tv_sec) * 1000000LL + (b.tv_usec - a.tv_usec);
    {ret}
}}"""
        pre = """static long long fake_d;
static int fake_calls;
static int fake_gettimeofday(struct timeval *t, void *tz) {
    long long v = (fake_calls++ & 1) ? fake_d : 0;
    (void)tz;
    t->tv_sec = (time_t)(v / 1000000LL);
    t->tv_usec = (suseconds_t)(v % 1000000LL);
    return 0;
}
#define gettimeofday fake_gettimeofday
"""
        name = "microseconds"
    body = _decoys(rng, body, {thr, thr + 1, work})
    main = (f'int main(void) {{\n    for (long long d = {thr - 1}LL; d <= {thr + 1}LL; d++) {{\n        fake_d = d;\n        fake_calls = 0;\n'
            '        printf("%d\\n", proc());\n    }\n    return 0;\n}')
    return Task(kind=FAMILY, variant=f"timing_{unit}", headers=HEADERS, body=body,
                question=question(f"above how many {name} between the two clock readings does it report a debugger? "
                                  f"Answer with a plain integer in {name}: no separators, no other unit. (The code may test `>= N + 1`; "
                                  f"N is the largest elapsed time it still lets pass.)"),
                truth=truth("timing", dec(thr), unit), params={"thr": thr, "unit": unit, "work": work, "form": form}, wrong=f"timing, {thr * 2}",
                check=(main, "0\n0\n1\n", pre), facts=[[thr, thr + 1, {"at": r"d > (\d+)LL", "kind": "near"}]])


def v_timing_ns(rng):
    return _timing(rng, "ns")


def v_timing_us(rng):
    return _timing(rng, "us")


def v_tracerpid(rng):
    rv = returns(rng)
    form = rng.randrange(3)
    size = rng.choice([128, 200, 256, 512])
    scan = ["""        if (strncmp(line, "TracerPid:", 10) == 0) {
            traced = atoi(line + 10);
            break;
        }""", """        if (sscanf(line, "TracerPid: %d", &traced) == 1) break;""", """        char *at = strstr(line, "TracerPid:");
        if (at) {
            traced = atoi(at + 10);
            break;
        }"""][form]
    ret = [f"return traced != 0 ? {rv} : 0;", f"if (traced) return {rv};\n    return 0;", f"return traced > 0 ? {rv} : 0;"][rng.randrange(3)]
    body = f"""int proc(void) {{
    char line[{size}];
    int traced = 0;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    while (fgets(line, sizeof line, f)) {{
{scan}
    }}
    fclose(f);
    {ret}
}}"""
    body = _decoys(rng, body, {rv, size})
    pre = """static char fake_buf[128];
static FILE *fake_fopen(const char *path, const char *mode) {
    (void)mode;
    if (strcmp(path, "/proc/self/status") != 0) return NULL;
    return fmemopen(fake_buf, strlen(fake_buf), "r");
}
#define fopen fake_fopen
"""
    main = ('int main(void) {\n    snprintf(fake_buf, sizeof fake_buf, "Name:\\tx\\nTracerPid:\\t0\\nUid:\\t1\\n");\n    printf("%d\\n", proc());\n'
            '    snprintf(fake_buf, sizeof fake_buf, "Name:\\tx\\nTracerPid:\\t4242\\nUid:\\t1\\n");\n    printf("%d\\n", proc());\n    return 0;\n}')
    return Task(kind=FAMILY, variant="tracerpid", headers=HEADERS, body=body,
                question=question("what does proc return when a tracer is attached (decimal)?"), truth=truth("tracerpid", dec(rv)),
                params={"rv": rv, "form": form, "size": size}, wrong=f"tracerpid, {rv + 1}", check=(main, f"0\n{rv}\n", pre),
                facts=[[rv, {"at": RV_AT, "kind": "exact"}]])


VARIANTS = {"ptrace": v_ptrace, "ptrace_raw": v_ptrace_raw, "env": v_env, "timing_ns": v_timing_ns, "timing_us": v_timing_us,
            "tracerpid": v_tracerpid}
