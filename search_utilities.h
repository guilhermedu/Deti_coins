#define next_value_to_try(v) \
    do { \
        v++; \
        if ((v & 0xFF) == 0x7F) { \
            v += 0xA1; /* 0x7F + 0xA1 = 0x120 */ \
            if (((v >> 8) & 0xFF) == 0x7F) { \
                v += 0xA1 << 8; \
                if (((v >> 16) & 0xFF) == 0x7F) { \
                    v += 0xA1 << 16; \
                    if (((v >> 24) & 0xFF) == 0x7F) { \
                        v += 0xA1 << 24; \
                    } \
                } \
            } \
        } \
    } while (0)

