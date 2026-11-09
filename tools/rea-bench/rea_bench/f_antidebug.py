"""Anti-debug techniques: name the technique and one detail read off the code.

Techniques: ptrace(PTRACE_TRACEME) through libc, the same through a raw syscall(SYS_ptrace, ...), a getenv() probe, a
clock_gettime / gettimeofday timing check, a TracerPid probe of /proc/self/status. The question lists the four technique words and
asks for one detail: the value returned on detection (ptrace, tracerpid), the environment variable (env), the threshold in
nanoseconds or microseconds (timing). The harness does not touch the real process: a preamble swaps the libc call for a fake
(`#define ptrace fake_ptrace`, a fake clock that moves by exactly D, a fake fopen on an in-memory status file), so the
test proves the request code, the syscall number, the path, the threshold and its unit, not just that the code compiles.
"""
from __future__ import annotations

from .core import HDR, Task, dec

FAMILY = "antidebug"
HEADERS = (HDR.rstrip("\n") + "\n#include <unistd.h>\n#include <time.h>\n#include <sys/time.h>\n#include <sys/ptrace.h>\n"
           "#include <sys/syscall.h>\n\n")

TECH = {"ptrace": ["ptrace", "ptrace_traceme", "traceme"], "timing": ["timing", "time-based", "clock_gettime", "gettimeofday", "rdtsc"],
        "env": ["env", "environment", "getenv"], "tracerpid": ["tracerpid"]}
OPTIONS = "one of: ptrace, timing, env, tracerpid"
RETURNS = [1, 2, 3, 7, 42, 99, 113, 255]
ENV_VARS = ["LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_LIBRARY_PATH", "LINES", "COLUMNS", "PYTHONINSPECT", "MALLOC_PERTURB_"]


def truth(tech, detail_alts):
    other = [n for t, ns in TECH.items() if t != tech for n in ns]
    return {"score": "tokens", "accept": [TECH[tech], detail_alts], "reject": other}


def question(detail):
    return f"Which anti-debugging technique does proc implement ({OPTIONS}), and {detail}"


def _ptrace(rng, raw):
    rv = rng.choice(RETURNS)
    form = rng.randrange(2)
    if raw:
        body = f"""int proc(void) {{
    long r = syscall(SYS_ptrace, PTRACE_TRACEME, 0, 0, 0);
    return r != 0 ? {rv} : 0;
}}"""
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
        body = (f"""int proc(void) {{
    long r = ptrace(PTRACE_TRACEME, 0, NULL, NULL);
    return r == -1 ? {rv} : 0;
}}""" if form == 0 else f"""int proc(void) {{
    if (ptrace(PTRACE_TRACEME, 0, 0, 0) < 0) {{
        return {rv};
    }}
    return 0;
}}""")
        pre = """static long fake_rv, last_req;
static long fake_ptrace(int req, ...) {
    last_req = req;
    return fake_rv;
}
#define ptrace fake_ptrace
"""
        proof = "    printf(\"%d\\n\", last_req == PTRACE_TRACEME);\n"
    main = (f'int main(void) {{\n    fake_rv = 0;\n    printf("%d\\n", proc());\n    fake_rv = -1;\n    printf("%d\\n", proc());\n{proof}    return 0;\n}}')
    pre = "#include <stdarg.h>\n" + pre
    return Task(kind=FAMILY, variant="ptrace_raw" if raw else "ptrace", headers=HEADERS, body=body,
                question=question("what does proc return when the call shows that a debugger is attached (decimal)?"),
                truth=truth("ptrace", dec(rv)), params={"rv": rv, "raw": raw}, wrong=f"timing, {rv}", check=(main, f"0\n{rv}\n1\n", pre))


def v_ptrace(rng):
    return _ptrace(rng, False)


def v_ptrace_raw(rng):
    return _ptrace(rng, True)


def v_env(rng):
    name, rv = rng.choice(ENV_VARS), rng.choice(RETURNS)
    if rng.randrange(2):
        body = f'int proc(void) {{\n    if (getenv("{name}") != NULL) return {rv};\n    return 0;\n}}'
    else:
        body = f'int proc(void) {{\n    const char *e = getenv("{name}");\n    if (e && *e) return {rv};\n    return 0;\n}}'
    main = (f'int main(void) {{\n    unsetenv("{name}");\n    printf("%d\\n", proc());\n    setenv("{name}", "1", 1);\n'
            f'    printf("%d\\n", proc());\n    return 0;\n}}')
    other = rng.choice([n for n in ENV_VARS if n != name])
    t = truth("env", [name.lower()])
    return Task(kind=FAMILY, variant="env", headers=HEADERS, body=body, question=question("which environment variable does it look at?"),
                truth=t, params={"name": name, "rv": rv}, wrong=f"env, {other}", check=(main, f"0\n{rv}\n"))


def _timing(rng, unit):
    ns = unit == "ns"
    k = rng.choice([10, 20, 50, 100, 250, 500])
    thr = k * 1_000_000 if ns else k * 1_000
    work = rng.choice([256, 1000, 4096])
    if ns:
        body = f"""int proc(void) {{
    struct timespec a, b;
    volatile unsigned x = 0;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i = 0; i < {work}; i++) x += (unsigned)i;
    clock_gettime(CLOCK_MONOTONIC, &b);
    long long d = (long long)(b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec);
    return d > {thr}LL;
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
    return d > {thr}LL;
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
    main = (f'int main(void) {{\n    for (long long d = {thr - 1}LL; d <= {thr + 1}LL; d++) {{\n        fake_d = d;\n        fake_calls = 0;\n'
            '        printf("%d\\n", proc());\n    }\n    return 0;\n}')
    return Task(kind=FAMILY, variant=f"timing_{unit}", headers=HEADERS, body=body,
                question=question(f"above how many {name} (decimal) between the two clock readings does it report a debugger?"),
                truth=truth("timing", dec(thr)), params={"thr": thr, "unit": unit, "work": work}, wrong=f"timing, {thr * 2}",
                check=(main, "0\n0\n1\n", pre))


def v_timing_ns(rng):
    return _timing(rng, "ns")


def v_timing_us(rng):
    return _timing(rng, "us")


def v_tracerpid(rng):
    rv = rng.choice(RETURNS)
    body = f"""int proc(void) {{
    char line[256];
    int traced = 0;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    while (fgets(line, sizeof line, f)) {{
        if (strncmp(line, "TracerPid:", 10) == 0) {{
            traced = atoi(line + 10);
            break;
        }}
    }}
    fclose(f);
    return traced != 0 ? {rv} : 0;
}}"""
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
                params={"rv": rv}, wrong=f"tracerpid, {rv + 1}", check=(main, f"0\n{rv}\n", pre))


VARIANTS = {"ptrace": v_ptrace, "ptrace_raw": v_ptrace_raw, "env": v_env, "timing_ns": v_timing_ns, "timing_us": v_timing_us,
            "tracerpid": v_tracerpid}
