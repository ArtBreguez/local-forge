/* Target of the phase 6 REA tool-use benchmark (issue #23, part C). Every task in ../tasks.json has its ground truth in THIS file.
 * Built on the host with TDM-GCC, exactly like tools/rea-target/checksum.c:
 *     C:\TDM-GCC-64\bin\gcc.exe -O1 -fno-lto -o bench.exe bench.c -lws2_32
 * The function names are neutral on purpose (rb_ = "rea bench"): what a function does is not in its name. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock2.h>

#define NOINLINE __attribute__((noinline))

/* integer finalizer ("lowbias32"): the first multiplier is 0x7feb352d */
NOINLINE unsigned rb_mix(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/* bitwise CRC-32, reflected polynomial 0xEDB88320 */
NOINLINE unsigned rb_crc(const unsigned char *p, int n) {
    unsigned c = 0xFFFFFFFFu;
    for (int i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

/* XOR every byte with one constant key: 0x5A */
NOINLINE void rb_unmask(unsigned char *p, int n) {
    for (int i = 0; i < n; i++) p[i] ^= 0x5A;
}

/* population count: clears the lowest set bit until nothing is left */
NOINLINE int rb_pop(unsigned x) {
    int n = 0;
    while (x) {
        x &= x - 1;
        n++;
    }
    return n;
}

/* Euclid's algorithm */
NOINLINE unsigned rb_gcd(unsigned a, unsigned b) {
    while (b) {
        unsigned t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* hard-coded credentials: user "admin", password "tr0ub4dor&3" */
NOINLINE int rb_login(const char *user, const char *pass) {
    if (strcmp(user, "admin") != 0) return 0;
    return strcmp(pass, "tr0ub4dor&3") == 0;
}

/* Fibonacci hashing: multiplies by 2^32 / golden ratio, 0x9E3779B9, keeps the top 12 bits */
NOINLINE unsigned rb_spread(unsigned x) {
    return (x * 0x9E3779B9u) >> 20;
}

/* iterative Fibonacci */
NOINLINE unsigned rb_fib(unsigned n) {
    unsigned a = 0, b = 1;
    while (n--) {
        unsigned t = a + b;
        a = b;
        b = t;
    }
    return a;
}

/* the only function that touches the network: opens a TCP socket and closes it again */
NOINLINE int rb_net(void) {
    WSADATA w;
    if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return -1;
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    int ok = s != INVALID_SOCKET;
    if (ok) closesocket(s);
    WSACleanup();
    return ok;
}

/* the only function that writes a file */
NOINLINE int rb_dump(const char *path, const void *buf, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(buf, 1, (size_t)n, f);
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    unsigned char data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const char *cmd = argc > 1 ? argv[1] : "all";
    if (!strcmp(cmd, "net")) return rb_net() < 0;
    if (!strcmp(cmd, "dump")) return rb_dump(argc > 2 ? argv[2] : "out.bin", data, (int)sizeof data);
    if (!strcmp(cmd, "login")) return !rb_login(argc > 2 ? argv[2] : "", argc > 3 ? argv[3] : "");
    rb_unmask(data, (int)sizeof data);
    printf("%u %u %d %u %u %u\n", rb_mix(0x12345678u), rb_crc(data, (int)sizeof data), rb_pop(0xF0F0u), rb_gcd(84u, 36u), rb_spread(1234u), rb_fib(20u));
    return 0;
}
