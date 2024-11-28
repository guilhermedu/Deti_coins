// deti_coins_client.h

#ifndef DETI_COINS_CLIENT_H
#define DETI_COINS_CLIENT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>

// Include necessary headers
#include "cpu_utilities.h"
#include "deti_coins_vault.h"

// Message Types (should match those defined in server code)
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
static int connect_to_server(char *ip_address, int port_number);
static void close_socket(int socket_fd);
static int send_message(int socket_fd, message_t *m);
static int receive_message(int socket_fd, message_t *m);
static void handle_received_coins(uint32_t (*coins)[13], uint32_t num_coins);
void deti_coins_client(char *server_address, int port_number);

// Implementations

void deti_coins_client(char *server_address, int port_number) {
    int client_socket = connect_to_server(server_address, port_number);

    while (1) {
        // Send work request
        message_t request = {MSG_WORK_REQUEST, 0, NULL};
        if (send_message(client_socket, &request) < 0) {
            perror("Failed to send work request");
            break;
        }

        // Receive results
        message_t response;
        if (receive_message(client_socket, &response) < 0) {
            perror("Failed to receive message");
            break;
        }

        if (response.msg_type == MSG_RESULTS) {
            // Process received coins
            uint32_t num_coins = response.payload_size / (13 * sizeof(uint32_t));
            uint32_t (*coins)[13] = (uint32_t (*)[13])response.payload;

            // Convert coins from network byte order
            for (uint32_t i = 0; i < num_coins; i++) {
                for (int j = 0; j < 13; j++) {
                    coins[i][j] = ntohl(coins[i][j]);
                }
            }

            // Handle the received coins
            handle_received_coins(coins, num_coins);

            // Send acknowledgment
            message_t ack_msg = {MSG_ACK, 0, NULL};
            if (send_message(client_socket, &ack_msg) < 0) {
                perror("Failed to send acknowledgment");
                break;
            }
        } else if (response.msg_type == MSG_NO_COINS_FOUND) {
            // No coins were found by the server
            printf("Server did not find any coins in this batch.\n");
        } else {
            // Unknown message type
            printf("Received unknown message type: %u\n", response.msg_type);
        }

        // Clean up
        if (response.payload_size > 0)
            free(response.payload);
    }

    // Close the connection
    close_socket(client_socket);
}

// Function to handle the received coins
static void handle_received_coins(uint32_t (*coins)[13], uint32_t num_coins) {
    for (uint32_t i = 0; i < num_coins; i++) {
        printf("Received coin from server:\n");
        char coin_text[53]; // 13 words * 4 bytes + null terminator
        int pos = 0;
        for (int j = 0; j < 13; j++) {
            uint32_t word = coins[i][j];
            // Extract bytes in correct order
            coin_text[pos++] = (word >> 24) & 0xFF;
            coin_text[pos++] = (word >> 16) & 0xFF;
            coin_text[pos++] = (word >> 8) & 0xFF;
            coin_text[pos++] = (word >> 0) & 0xFF;
        }
        coin_text[pos] = '\0';
        printf("%s\n", coin_text);

        // Optionally, save the coin using save_deti_coin()
        save_deti_coin(coins[i]);
    }
}


// Function to connect to the server
static int connect_to_server(char *ip_address, int port_number) {
    struct sockaddr_in server_addr;
    int connection_fd;

    connection_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (connection_fd == -1) {
        perror("connect_to_server(): socket");
        exit(1);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_number);
    if (inet_pton(AF_INET, ip_address, &server_addr.sin_addr) <= 0) {
        perror("connect_to_server(): inet_pton");
        exit(1);
    }

    if (connect(connection_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("connect_to_server(): connect");
        exit(1);
    }

    printf("Connected to server at %s:%d\n", ip_address, port_number);
    return connection_fd;
}

// Function to close a socket
static void close_socket(int socket_fd) {
    if (close(socket_fd) < 0) {
        perror("close_socket(): close");
        exit(1);
    }
}

// Communication functions
static int send_message(int socket_fd, message_t *m) {
    uint32_t header[2];
    header[0] = htonl(m->msg_type);
    header[1] = htonl(m->payload_size);

    // Send header
    if (send(socket_fd, header, sizeof(header), 0) != sizeof(header)) {
        perror("send_message(): send header");
        return -1;
    }

    // Send payload if there is any
    if (m->payload_size > 0) {
        if (send(socket_fd, m->payload, m->payload_size, 0) != (ssize_t)m->payload_size) {
            perror("send_message(): send payload");
            return -1;
        }
    }

    return 0;
}

static int receive_message(int socket_fd, message_t *m) {
    uint32_t header[2];

    // Receive header
    if (recv(socket_fd, header, sizeof(header), MSG_WAITALL) != sizeof(header)) {
        perror("receive_message(): recv header");
        return -1;
    }

    m->msg_type = ntohl(header[0]);
    m->payload_size = ntohl(header[1]);

    // Allocate memory for payload
    if (m->payload_size > 0) {
        m->payload = malloc(m->payload_size);
        if (m->payload == NULL) {
            perror("receive_message(): malloc payload");
            return -1;
        }

        // Receive payload
        if (recv(socket_fd, m->payload, m->payload_size, MSG_WAITALL) != (ssize_t)m->payload_size) {
            perror("receive_message(): recv payload");
            free(m->payload);
            return -1;
        }
    } else {
        m->payload = NULL;
    }

    return 0;
}

#endif // DETI_COINS_CLIENT_H
