// deti_coins_server.h

#ifndef DETI_COINS_SERVER_AVX2_H
#define DETI_COINS_SERVER_AVX2_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// Include your custom headers
#include "md5_cpu_avx2.h"
#include "cpu_utilities.h"
#include "deti_coins_vault.h"
#include "deti_coins_cpu_avx2_SIMD_OPENMP.h"


// No need for stop_request in the server logic
// Remove any dependency on global stop_request

#define PORT_NUMBER 5000
#define BACKLOG 10
#define INITIAL_SEED 123456789

#define MAX_COINS_PER_REQUEST 4

// Message Types
#define MSG_WORK_REQUEST    1
#define MSG_RESULTS         3
#define MSG_NO_COINS_FOUND  6
#define MSG_ACK             5

typedef struct {
    uint32_t msg_type;
    uint32_t payload_size;
    void *payload; // Pointer to the actual data
} message_t;

// Function prototypes
static int setup_server(int port_number);
static int get_connection(int listen_fd, char connection_ipv4_address[32]);
static void close_socket(int socket_fd);
static int send_message(int socket_fd, message_t *m);
static int receive_message(int socket_fd, message_t *m);
static void *handle_client(void *arg);
static void perform_computation(uint32_t seed, uint32_t coins_found[][13], int *num_coins_found);
static uint32_t get_next_seed();

// Global variables
static pthread_mutex_t seed_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint32_t current_seed = INITIAL_SEED;

// Function to get the next seed
static uint32_t get_next_seed() {
    pthread_mutex_lock(&seed_mutex);
    uint32_t seed = current_seed++;
    pthread_mutex_unlock(&seed_mutex);
    return seed;
}

void deti_coins_cpu_avx2_server(uint32_t seed, uint32_t coins_found[][13], int *num_coins_found, int max_coins) {
    uint64_t n_attempts = 0;
    *num_coins_found = 0;

    #pragma omp parallel reduction(+:n_attempts)
    {
        __m256i xorshift32_state_avx2[NUM_VARIABLES];
        uint32_t data[13][NUM_LANES] __attribute__((aligned(32)));
        uint32_t hash[4][NUM_LANES];
        uint32_t v1[NUM_LANES], v2[NUM_LANES], v3[NUM_LANES], v4[NUM_LANES];
        uint32_t coin[13];

        // Initialize PRNG state with improved seeding
        unsigned int thread_num = omp_get_thread_num();
        unsigned int time_seed = (unsigned int)time(NULL);
        unsigned int prng_seed = seed ^ time_seed ^ thread_num;

        unsigned int rand_seed = prng_seed;
        for (int var = 0; var < NUM_VARIABLES; var++) {
            uint32_t lane_seeds[NUM_LANES];
            for (int lane = 0; lane < NUM_LANES; lane++) {
                // Use rand_r() for thread-safe random numbers
                lane_seeds[lane] = prng_seed + var * 997 + lane * 12345 + thread_num * 54321 + rand_r(&rand_seed);
            }
            xorshift32_state_avx2[var] = _mm256_set_epi32(
                lane_seeds[7],
                lane_seeds[6],
                lane_seeds[5],
                lane_seeds[4],
                lane_seeds[3],
                lane_seeds[2],
                lane_seeds[1],
                lane_seeds[0]
            );
        }

        // Initialize data array
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < NUM_LANES; j++) {
                data[i][j] = 0x20202020; // Little-endian '    '
            }
        }

        // Set fixed parts of the coin
        for (int j = 0; j < NUM_LANES; j++) {
            data[0][j] = 0x49544544; // "DETI" in little-endian
            data[1][j] = 0x696F6320; // "coi " in little-endian
            data[2][j] = 0x6E20206E; // "n  n" in little-endian
            data[12][j] = 0x0A202020; // Newline character '\n' followed by spaces
        }

        // Local variable to control loop termination
        int local_stop = 0;

        while (*num_coins_found < max_coins && local_stop == 0) {
            random_printable_u32_avx2_server(v1, &xorshift32_state_avx2[0]);
            random_printable_u32_avx2_server(v2, &xorshift32_state_avx2[1]);
            random_printable_u32_avx2_server(v3, &xorshift32_state_avx2[2]);
            random_printable_u32_avx2_server(v4, &xorshift32_state_avx2[3]);

            for (int i = 0; i < NUM_LANES; i++) {
                data[3][i] = v1[i];
                data[4][i] = v2[i];
                data[5][i] = v3[i];
                data[6][i] = v4[i];
            }

            md5_cpu_avx2((v8si *)data, (v8si *)hash);

            for (int lane = 0; lane < NUM_LANES; lane++) {
                uint32_t hash_lane[4];

                for (int i = 0; i < 4; i++) {
                    hash_lane[i] = hash[i][lane];
                }

                hash_byte_reverse(hash_lane);

                uint32_t n = deti_coin_power(hash_lane);

                if (n >= 31){
                    printf("Coin with %d zeros found\n", n);
                    printf("Hash: %x %x %x %x\n", hash_lane[0], hash_lane[1], hash_lane[2], hash_lane[3]);
                }

                if (n >= 32u) {
                    for (int i = 0; i < 13; i++) {
                        coin[i] = data[i][lane];
                    }
                    #pragma omp critical
                    {
                        if (*num_coins_found < max_coins) {
                            // Store the coin
                            for (int i = 0; i < 13; i++) {
                                coins_found[*num_coins_found][i] = coin[i];
                            }
                            (*num_coins_found)++;
                            save_deti_coin(coin);

                            // Check if we've reached the maximum number of coins
                            if (*num_coins_found >= max_coins) {
                                local_stop = 1;
                            }
                        }
                    }
                }
            }

            n_attempts += NUM_LANES;
        }
    }
    STORE_DETI_COINS();
    printf("deti_coins_cpu_avx2_simd_openmp_search: %lu DETI coin%s found in %lu attempt%s (expected %.2f coins)\n",
           *num_coins_found, (*num_coins_found == 1ul) ? "" : "s", 
           n_attempts, (n_attempts == 1ul) ? "" : "s",
           (double)n_attempts / (double)(1ul << 32));
}

// Helper functions

static inline __m256i xorshift32_avx2_server(__m256i *state) {
    __m256i x = *state;

    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));

    *state = x;
    return x;
}

void random_printable_u32_avx2_server(uint32_t *v, __m256i *state) {
    __m256i rand_nums = xorshift32_avx2_server(state);

    // Extract bytes from rand_nums
    __m256i rand_bytes = rand_nums;

    // Unpack 8-bit integers to 16-bit integers
    __m256i zero = _mm256_setzero_si256();
    __m256i rand_bytes_lo = _mm256_unpacklo_epi8(rand_bytes, zero);
    __m256i rand_bytes_hi = _mm256_unpackhi_epi8(rand_bytes, zero);

    // Define mask and base for printable ASCII characters
    __m256i mask = _mm256_set1_epi16(95); // Range size (0x7E - 0x20 + 1)
    __m256i base = _mm256_set1_epi16(0x20); // Base ASCII value (32)

    // Use modulo operation (since AVX2 doesn't have modulo, use bitwise AND if the range is power of two)
    // For 95, we can't use bitwise AND, so we need to simulate modulo or limit values

    // Alternative: Use `_mm256_rem_epi16` equivalent or adjust method
    // For simplicity, we'll use bitmasking to limit the range, acknowledging it's not perfect

    // Apply mask to limit the range (this will produce values from 0 to 94)
    __m256i chars_lo = _mm256_and_si256(rand_bytes_lo, _mm256_set1_epi16(0x7F));
    __m256i chars_hi = _mm256_and_si256(rand_bytes_hi, _mm256_set1_epi16(0x7F));

    // Add base to shift to the printable ASCII range
    chars_lo = _mm256_add_epi16(chars_lo, base);
    chars_hi = _mm256_add_epi16(chars_hi, base);

    // Pack back to 8-bit integers
    __m256i chars = _mm256_packus_epi16(chars_lo, chars_hi);

    // Store the result
    _mm256_storeu_si256((__m256i *)v, chars);
}



// Main server function
void deti_coins_server_avx2() {
    int listen_fd;

    // Set up the server socket
    listen_fd = setup_server(PORT_NUMBER);

    printf("Server listening on port %d\n", PORT_NUMBER);

    while (1) {
        // Accept new client connection
        int *client_socket = malloc(sizeof(int));
        if (!client_socket) {
            perror("Failed to allocate memory for client socket");
            continue;
        }
        *client_socket = get_connection(listen_fd, NULL);

        // Create a new thread to handle the client
        pthread_t thread_id;
        if (pthread_create(&thread_id, NULL, handle_client, (void *)client_socket) != 0) {
            perror("Failed to create thread");
            close_socket(*client_socket);
            free(client_socket);
            continue;
        }
        pthread_detach(thread_id);
    }

    // Close the listening socket (unreachable code in this loop)
    close_socket(listen_fd);
}

// Function to handle client communication
static void *handle_client(void *arg) {
    int client_socket = *(int *)arg;
    free(arg);

    message_t m;
    int keep_alive = 1;
    while (keep_alive) {
        if (receive_message(client_socket, &m) < 0) {
            // Handle error or client disconnect
            break;
        }

        // Process the message based on its type
        switch (m.msg_type) {
            case MSG_WORK_REQUEST: {
                // Perform computation using the existing function
                uint32_t seed = get_next_seed();

                printf("Performing computation for client with seed: %u\n", seed);

                // Prepare to store coins found
                uint32_t coins_found[MAX_COINS_PER_REQUEST][13];
                int num_coins_found = 0;

                // Perform the computation
                perform_computation(seed, coins_found, &num_coins_found);

                // Send results back to client
                if (num_coins_found > 0) {
                    // Convert coins to network byte order
                    for (int i = 0; i < num_coins_found; i++) {
                        for (int j = 0; j < 13; j++) {
                            coins_found[i][j] = htonl(coins_found[i][j]);
                        }
                    }

                    message_t results_msg;
                    results_msg.msg_type = MSG_RESULTS;
                    results_msg.payload_size = num_coins_found * 13 * sizeof(uint32_t);
                    results_msg.payload = coins_found;

                    send_message(client_socket, &results_msg);

                    // Optionally receive acknowledgment
                    message_t ack;
                    if (receive_message(client_socket, &ack) < 0) {
                        // Handle error
                        break;
                    }
                    // Clean up
                    if (ack.payload_size > 0)
                        free(ack.payload);
                } else {
                    // Send a message indicating no coins were found
                    message_t no_coins_msg = {MSG_NO_COINS_FOUND, 0, NULL};
                    send_message(client_socket, &no_coins_msg);
                }

                break;
            }
            default:
                // Unknown message type
                printf("Received unknown message type.\n");
                keep_alive = 0;
                break;
        }

        // Clean up payload
        if (m.payload_size > 0) {
            free(m.payload);
        }
    }

    close_socket(client_socket);
    return NULL;
}

// Implement the computation function
static void perform_computation(uint32_t seed, uint32_t coins_found[][13], int *num_coins_found) {
    // Set a limit for the number of coins to find
    int max_coins = MAX_COINS_PER_REQUEST;

    // Call your AVX2 search function
    deti_coins_cpu_avx2_server(seed, coins_found, num_coins_found, max_coins);

    // After computation, the coins_found array and num_coins_found will be updated
}

// Function to set up the server socket
static int setup_server(int port_number) {
    struct sockaddr_in server_addr;
    int listen_fd;
    int opt = 1;

    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == -1) {
        perror("setup_server(): socket");
        exit(1);
    }

    // Set socket options to reuse the address and port
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setup_server(): setsockopt");
        exit(1);
    }

    memset(&server_addr, 0, sizeof(server_addr)); // Clear structure
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_number);

    if (bind(listen_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("setup_server(): bind");
        exit(1);
    }

    if (listen(listen_fd, BACKLOG) == -1) {
        perror("setup_server(): listen");
        exit(1);
    }

    return listen_fd;
}

// Function to accept a new client connection
static int get_connection(int listen_fd, char connection_ipv4_address[32]) {
    struct sockaddr_in peer_address;
    socklen_t peer_length;
    int connection_fd;

    peer_length = sizeof(peer_address);
    connection_fd = accept(listen_fd, (struct sockaddr *)&peer_address, &peer_length);
    if (connection_fd < 0) {
        perror("get_connection(): accept");
        exit(1);
    }
    if (connection_ipv4_address != NULL)
        snprintf(connection_ipv4_address, 32, "%s", inet_ntoa(peer_address.sin_addr));
    return connection_fd;
}

// Function to close a socket
static void close_socket(int socket_fd) {
    if (close(socket_fd) < 0) {
        perror("close_socket(): close");
    }
}

// Communication functions
static int send_message(int socket_fd, message_t *m) {
    uint32_t header[2];
    header[0] = htonl(m->msg_type);
    header[1] = htonl(m->payload_size);

    // Send header
    if (send(socket_fd, header, sizeof(header), 0) != sizeof(header))
        return -1;

    // Send payload if there is any
    if (m->payload_size > 0) {
        if (send(socket_fd, m->payload, m->payload_size, 0) != (ssize_t)m->payload_size)
            return -1;
    }

    return 0;
}

static int receive_message(int socket_fd, message_t *m) {
    uint32_t header[2];

    // Receive header
    if (recv(socket_fd, header, sizeof(header), MSG_WAITALL) != sizeof(header))
        return -1;

    m->msg_type = ntohl(header[0]);
    m->payload_size = ntohl(header[1]);

    // Allocate memory for payload
    if (m->payload_size > 0) {
        m->payload = malloc(m->payload_size);
        if (m->payload == NULL)
            return -1;

        // Receive payload
        if (recv(socket_fd, m->payload, m->payload_size, MSG_WAITALL) != (ssize_t)m->payload_size) {
            free(m->payload);
            return -1;
        }
    } else {
        m->payload = NULL;
    }

    return 0;
}

#endif // DETI_COINS_SERVER_H
