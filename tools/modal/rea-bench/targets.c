/* local-forge REA benchmark targets. Each target is a small program with a known answer; the ground truth lives in
   rea-tasks.json next to this file and is written before any model sees the binaries. Compiled with -O1, stripped or not
   per task. Nothing here is copied from anywhere. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* T1: what does this compute? (FNV-1a 32-bit with a twist: xor with 0x5bd1e995 at the end) */
uint32_t tag_a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) { h ^= (uint8_t)*s; h *= 16777619u; }
    return h ^ 0x5bd1e995u;
}

/* T2: recover the key. XOR cipher with the repeating key "m4gn0lia" */
void tag_b(uint8_t *buf, size_t n) {
    static const char key[] = "m4gn0lia";
    for (size_t i = 0; i < n; i++) buf[i] ^= (uint8_t)key[i % 8];
}

/* T3: what is the magic number the file header must start with? (0x4C465247 = "GRFL" little-endian) */
int tag_c(const uint8_t *hdr, size_t n) {
    if (n < 8) return -1;
    uint32_t magic; memcpy(&magic, hdr, 4);
    if (magic != 0x4C465247u) return -2;
    uint32_t ver; memcpy(&ver, hdr + 4, 4);
    return ver > 3 ? -3 : (int)ver;
}

/* T4: which password unlocks it? constant-time compare against "orchid-7781" */
int tag_d(const char *pw) {
    static const char want[] = "orchid-7781";
    size_t n = strlen(want);
    if (strlen(pw) != n) return 0;
    unsigned d = 0;
    for (size_t i = 0; i < n; i++) d |= (unsigned)(pw[i] ^ want[i]);
    return d == 0;
}

/* T5: how many iterations, and what is the modulus? (LCG, a=1103515245, c=12345, m=2^31, 17 rounds) */
uint32_t tag_e(uint32_t seed) {
    for (int i = 0; i < 17; i++) seed = (seed * 1103515245u + 12345u) & 0x7fffffffu;
    return seed;
}

/* T6: what does the checksum skip? (sums every byte except ones equal to 0x20, then adds the length) */
uint32_t tag_f(const uint8_t *p, size_t n) {
    uint32_t s = 0;
    for (size_t i = 0; i < n; i++) if (p[i] != 0x20) s += p[i];
    return s + (uint32_t)n;
}

/* T7: what format is this parsing? (key=value lines, '#' comments, returns count of keys named "port") */
int tag_g(const char *text) {
    int ports = 0; const char *line = text;
    while (*line) {
        const char *nl = strchr(line, '\n'); size_t len = nl ? (size_t)(nl - line) : strlen(line);
        if (len > 5 && line[0] != '#' && strncmp(line, "port=", 5) == 0) ports++;
        line += len + (nl ? 1 : 0);
    }
    return ports;
}

/* T8: what is the base of this encoding? (custom base32 alphabet, returns alphabet index of a char) */
int tag_h(char c) {
    static const char alpha[] = "ybndrfg8ejkmcpqxot1uwisza345h769";  /* z-base-32 */
    const char *p = strchr(alpha, c);
    return p ? (int)(p - alpha) : -1;
}

int main(int argc, char **argv) {
    const char *in = argc > 1 ? argv[1] : "local-forge";
    uint8_t buf[64]; size_t n = strlen(in); if (n > 64) n = 64; memcpy(buf, in, n); tag_b(buf, n);
    printf("%u %d %d %u %u %d %d\n", tag_a(in), tag_c((const uint8_t *)in, n), tag_d(in), tag_e(7), tag_f(buf, n),
           tag_g(in), tag_h(in[0]));
    return 0;
}
