// deti_coins_opencl_kernel.cl

// Define necessary macros
#define C(c)         (c)
#define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
#define DATA(idx)    coin[idx]
#define HASH(idx)    hash[idx]
#define STATE(idx)   state[idx]
#define X(idx)       x[idx]
#define MD5_OP(F,a,b,c,d,x,s,ac)  a += F(b,c,d) + x + C(ac); if(s != 0) a = ROTATE(a,s); a += b

// Define MD5 auxiliary functions
#define MD5_F(x,y,z) (((x) & (y)) | ((~(x)) & (z)))
#define MD5_G(x,y,z) (((x) & (z)) | ((y) & (~(z))))
#define MD5_H(x,y,z) ((x) ^ (y) ^ (z))
#define MD5_I(x,y,z) ((y) ^ ((x) | (~(z))))

// Define shift amounts
#define MD5_11 7
#define MD5_12 12
#define MD5_13 17
#define MD5_14 22
#define MD5_21 5
#define MD5_22 9
#define MD5_23 14
#define MD5_24 20
#define MD5_31 4
#define MD5_32 11
#define MD5_33 16
#define MD5_34 23
#define MD5_41 6
#define MD5_42 10
#define MD5_43 15
#define MD5_44 21

// Define CUSTOM_MD5_CODE macro
#define CUSTOM_MD5_CODE()                           \
  do                                                \
  {                                                 \
    /* initial state */                             \
    STATE(0) = C(0x67452301u);                      \
    STATE(1) = C(0xEFCDAB89u);                      \
    STATE(2) = C(0x98BADCFEu);                      \
    STATE(3) = C(0x10325476u);                      \
    a = STATE(0);                                   \
    b = STATE(1);                                   \
    c = STATE(2);                                   \
    d = STATE(3);                                   \
    /* initial data (13*4 bytes) + padding */       \
    X( 0) = DATA( 0);                               \
    X( 1) = DATA( 1);                               \
    X( 2) = DATA( 2);                               \
    X( 3) = DATA( 3);                               \
    X( 4) = DATA( 4);                               \
    X( 5) = DATA( 5);                               \
    X( 6) = DATA( 6);                               \
    X( 7) = DATA( 7);                               \
    X( 8) = DATA( 8);                               \
    X( 9) = DATA( 9);                               \
    X(10) = DATA(10);                               \
    X(11) = DATA(11);                               \
    X(12) = DATA(12);                               \
    X(13) = C(0x00000080u); /* padding */           \
    X(14) = C(13u * 32u);   /* number  */           \
    X(15) = C(0x00000000u); /* of bits */           \
    /* first round */                               \
    MD5_OP(MD5_F,a,b,c,d,X( 0),MD5_11,0xD76AA478u); \
    MD5_OP(MD5_F,d,a,b,c,X( 1),MD5_12,0xE8C7B756u); \
    MD5_OP(MD5_F,c,d,a,b,X( 2),MD5_13,0x242070DBu); \
    MD5_OP(MD5_F,b,c,d,a,X( 3),MD5_14,0xC1BDCEEEu); \
    MD5_OP(MD5_F,a,b,c,d,X( 4),MD5_11,0xF57C0FAFu); \
    MD5_OP(MD5_F,d,a,b,c,X( 5),MD5_12,0x4787C62Au); \
    MD5_OP(MD5_F,c,d,a,b,X( 6),MD5_13,0xA8304613u); \
    MD5_OP(MD5_F,b,c,d,a,X( 7),MD5_14,0xFD469501u); \
    MD5_OP(MD5_F,a,b,c,d,X( 8),MD5_11,0x698098D8u); \
    MD5_OP(MD5_F,d,a,b,c,X( 9),MD5_12,0x8B44F7AFu); \
    MD5_OP(MD5_F,c,d,a,b,X(10),MD5_13,0xFFFF5BB1u); \
    MD5_OP(MD5_F,b,c,d,a,X(11),MD5_14,0x895CD7BEu); \
    MD5_OP(MD5_F,a,b,c,d,X(12),MD5_11,0x6B901122u); \
    MD5_OP(MD5_F,d,a,b,c,X(13),MD5_12,0xFD987193u); \
    MD5_OP(MD5_F,c,d,a,b,X(14),MD5_13,0xA679438Eu); \
    MD5_OP(MD5_F,b,c,d,a,X(15),MD5_14,0x49B40821u); \
    /* second round */                              \
    MD5_OP(MD5_G,a,b,c,d,X( 1),MD5_21,0xF61E2562u); \
    MD5_OP(MD5_G,d,a,b,c,X( 6),MD5_22,0xC040B340u); \
    MD5_OP(MD5_G,c,d,a,b,X(11),MD5_23,0x265E5A51u); \
    MD5_OP(MD5_G,b,c,d,a,X( 0),MD5_24,0xE9B6C7AAu); \
    MD5_OP(MD5_G,a,b,c,d,X( 5),MD5_21,0xD62F105Du); \
    MD5_OP(MD5_G,d,a,b,c,X(10),MD5_22,0x02441453u); \
    MD5_OP(MD5_G,c,d,a,b,X(15),MD5_23,0xD8A1E681u); \
    MD5_OP(MD5_G,b,c,d,a,X( 4),MD5_24,0xE7D3FBC8u); \
    MD5_OP(MD5_G,a,b,c,d,X( 9),MD5_21,0x21E1CDE6u); \
    MD5_OP(MD5_G,d,a,b,c,X(14),MD5_22,0xC33707D6u); \
    MD5_OP(MD5_G,c,d,a,b,X( 3),MD5_23,0xF4D50D87u); \
    MD5_OP(MD5_G,b,c,d,a,X( 8),MD5_24,0x455A14EDu); \
    MD5_OP(MD5_G,a,b,c,d,X(13),MD5_21,0xA9E3E905u); \
    MD5_OP(MD5_G,d,a,b,c,X( 2),MD5_22,0xFCEFA3F8u); \
    MD5_OP(MD5_G,c,d,a,b,X( 7),MD5_23,0x676F02D9u); \
    MD5_OP(MD5_G,b,c,d,a,X(12),MD5_24,0x8D2A4C8Au); \
    /* third round */                               \
    MD5_OP(MD5_H,a,b,c,d,X( 5),MD5_31,0xFFFA3942u); \
    MD5_OP(MD5_H,d,a,b,c,X( 8),MD5_32,0x8771F681u); \
    MD5_OP(MD5_H,c,d,a,b,X(11),MD5_33,0x6D9D6122u); \
    MD5_OP(MD5_H,b,c,d,a,X(14),MD5_34,0xFDE5380Cu); \
    MD5_OP(MD5_H,a,b,c,d,X( 1),MD5_31,0xA4BEEA44u); \
    MD5_OP(MD5_H,d,a,b,c,X( 4),MD5_32,0x4BDECFA9u); \
    MD5_OP(MD5_H,c,d,a,b,X( 7),MD5_33,0xF6BB4B60u); \
    MD5_OP(MD5_H,b,c,d,a,X(10),MD5_34,0xBEBFBC70u); \
    MD5_OP(MD5_H,a,b,c,d,X(13),MD5_31,0x289B7EC6u); \
    MD5_OP(MD5_H,d,a,b,c,X( 0),MD5_32,0xEAA127FAu); \
    MD5_OP(MD5_H,c,d,a,b,X( 3),MD5_33,0xD4EF3085u); \
    MD5_OP(MD5_H,b,c,d,a,X( 6),MD5_34,0x04881D05u); \
    MD5_OP(MD5_H,a,b,c,d,X( 9),MD5_31,0xD9D4D039u); \
    MD5_OP(MD5_H,d,a,b,c,X(12),MD5_32,0xE6DB99E5u); \
    MD5_OP(MD5_H,c,d,a,b,X(15),MD5_33,0x1FA27CF8u); \
    MD5_OP(MD5_H,b,c,d,a,X( 2),MD5_34,0xC4AC5665u); \
    /* fourth round */                              \
    MD5_OP(MD5_I,a,b,c,d,X( 0),MD5_41,0xF4292244u); \
    MD5_OP(MD5_I,d,a,b,c,X( 7),MD5_42,0x432AFF97u); \
    MD5_OP(MD5_I,c,d,a,b,X(14),MD5_43,0xAB9423A7u); \
    MD5_OP(MD5_I,b,c,d,a,X( 5),MD5_44,0xFC93A039u); \
    MD5_OP(MD5_I,a,b,c,d,X(12),MD5_41,0x655B59C3u); \
    MD5_OP(MD5_I,d,a,b,c,X( 3),MD5_42,0x8F0CCC92u); \
    MD5_OP(MD5_I,c,d,a,b,X(10),MD5_43,0xFFEFF47Du); \
    MD5_OP(MD5_I,b,c,d,a,X( 1),MD5_44,0x85845DD1u); \
    MD5_OP(MD5_I,a,b,c,d,X( 8),MD5_41,0x6FA87E4Fu); \
    MD5_OP(MD5_I,d,a,b,c,X(15),MD5_42,0xFE2CE6E0u); \
    MD5_OP(MD5_I,c,d,a,b,X( 6),MD5_43,0xA3014314u); \
    MD5_OP(MD5_I,b,c,d,a,X(13),MD5_44,0x4E0811A1u); \
    MD5_OP(MD5_I,a,b,c,d,X( 4),MD5_41,0xF7537E82u); \
    MD5_OP(MD5_I,d,a,b,c,X(11),MD5_42,0xBD3AF235u); \
    MD5_OP(MD5_I,c,d,a,b,X( 2),MD5_43,0x2AD7D2BBu); \
    MD5_OP(MD5_I,b,c,d,a,X( 9),MD5_44,0xEB86D391u); \
    /* update state */                              \
    STATE(0) += a;                                  \
    STATE(1) += b;                                  \
    STATE(2) += c;                                  \
    STATE(3) += d;                                  \
    /* record hash value */                         \
    HASH(0) = STATE(0);                             \
    HASH(1) = STATE(1);                             \
    HASH(2) = STATE(2);                             \
    HASH(3) = STATE(3);                             \
  }                                                 \
  while(0)


// Function to increment a word while keeping characters printable
void increment_coin_word(uint *word) {
    for (int i = 0; i < 4; ++i) {
        uchar c = (uchar)((*word >> (i * 8)) & 0xFFu);
        if (c < (uchar)0x7Eu) {
            c++;
            *word = (*word & ~(0xFFu << (i * 8))) | ((uint)c << (i * 8));
            return;
        } else {
            c = (uchar)0x20u;
            *word = (*word & ~(0xFFu << (i * 8))) | ((uint)c << (i * 8));
        }
    }
}

// Start of kernel function
__kernel void deti_coins_opencl_kernel_search(
   __global uint *deti_coins_storage_area,
   uint random_word,
   uint custom_word_1,
   uint custom_word_2)
{
   uint n = get_global_id(0);
   uint coin[13];
   uint hash[4];
   uint a, b, c, d;
   uint state[4], x[16];

   // Initialize DETI coin template
   coin[0] = 0x49544544u; // 'DETI' in little-endian
   coin[1] = 0x696F6320u; // 'coi '
   coin[2] = 0x6E20206Eu; // 'n  n'
   coin[3] = random_word;
   coin[4] = custom_word_1;
   coin[5] = custom_word_2;

   // Properly initialize coin[6] based on n
   coin[6] = 0;
   uint temp_n = n;
   for (int shift = 0; shift <= 24; shift += 8) {
       uchar char_code = (uchar)((temp_n % (0x7E - 0x20 + 1)) + 0x20u);
       coin[6] |= ((uint)char_code) << shift;
       temp_n /= (0x7E - 0x20 + 1);
   }

   // Properly initialize coin[7] based on n
   coin[7] = 0;
   temp_n = n + 12345u; // Offset to vary starting points
   for (int shift = 0; shift <= 24; shift += 8) {
       uchar char_code = (uchar)((temp_n % (0x7E - 0x20 + 1)) + 0x20u);
       coin[7] |= ((uint)char_code) << shift;
       temp_n /= (0x7E - 0x20 + 1);
   }

   // Initialize the rest of the coin
   for (int i = 8; i < 12; i++) {
       coin[i] = 0x20202020u; // Padding with spaces
   }
   coin[12] = 0x0A202020u; // Ending with newline and spaces

   // Start generating coins by incrementing multiple words
   for (uint i = 0; i < 256u; i++) {
       increment_coin_word(&coin[7]);
       increment_coin_word(&coin[8]);
       increment_coin_word(&coin[9]);

       // Compute MD5 hash
       CUSTOM_MD5_CODE();

       // If hash meets criteria, store the coin
       if (hash[3] == 0u) {
           uint index = atomic_add(&deti_coins_storage_area[0], 13u);
           if (index <= 1024u - 13u) {
               for (int j = 0; j < 13; j++) {
                   deti_coins_storage_area[index + j] = coin[j];
               }
           }
       }
   }
}
