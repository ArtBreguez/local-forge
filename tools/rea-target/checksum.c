/* Known-source test target for REA + Ghidra decompilation. */
#include <stdio.h>
#include <stdlib.h>

__declspec(noinline) int forge_checksum(const unsigned char *buf, int len) {
    unsigned int acc = 0x1505u;
    for (int i = 0; i < len; i++) {
        acc = ((acc << 5) + acc) ^ buf[i];   /* djb2-xor */
        if (buf[i] == 0x2A) acc += 7;
    }
    return (int)(acc & 0x7FFFFFFF);
}

int main(int argc, char **argv) {
    const char *s = argc > 1 ? argv[1] : "local-forge";
    int n = 0;
    while (s[n]) n++;
    printf("%d\n", forge_checksum((const unsigned char *)s, n));
    return 0;
}
