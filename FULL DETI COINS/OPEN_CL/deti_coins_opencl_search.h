#ifndef DETI_COINS_OPENCL_SEARCH_H
#define DETI_COINS_OPENCL_SEARCH_H

// Define OpenCL version to avoid warnings
#define CL_TARGET_OPENCL_VERSION 200

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "search_utilities.h"
#include <time.h>
#include <unistd.h>

extern volatile int stop_request;

// Gera um valor aleatório de 32 bits com caracteres ASCII imprimíveis
static uint32_t random_value_to_try_ascii() {
    uint32_t value = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t random_char = (uint8_t)(0x20 + (rand() % (0x7F - 0x20)));
        value |= ((uint32_t)random_char) << (i * 8);
    }
    return value;
}

void md5_compute(const uint32_t *data, size_t size, uint32_t *hash) {
    uint32_t a,b,c,d,state[4],x[16];
    # define C(c)         (c)
    # define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
    # define DATA(idx)    data[idx]
    # define HASH(idx)    hash[idx]
    # define STATE(idx)   state[idx]
    # define X(idx)       x[idx]
    CUSTOM_MD5_CODE();
    # undef C
    # undef ROTATE
    # undef DATA
    # undef HASH
    # undef STATE
    # undef X
}



static void deti_coins_opencl_search(uint32_t n_random_words)
{
    uint32_t idx, max_idx, random_word, custom_word_1, custom_word_2;
    uint64_t n_attempts = 0ul, n_coins = 0ul;
    size_t global_work_size = (1u << 20); // Total de work-items
    size_t local_work_size = 128; // Tamanho do work-group
    max_idx = 1u;
    cl_int err;

    random_word = (n_random_words == 0u) ? 0x20202020u : random_value_to_try_ascii();
    // Inicializa palavras customizadas
    custom_word_1 = custom_word_2 = 0x20202020u;

    // Inicializa o OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem device_data = NULL;

    // Carrega o código do kernel OpenCL a partir do arquivo
    const char *filename = "deti_coins_opencl_kernel.cl";
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel file %s.\n", filename);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t kernel_size = ftell(fp);
    rewind(fp);
    char *kernel_source = (char *)malloc(kernel_size + 1);
    if (!kernel_source) {
        fprintf(stderr, "Failed to allocate memory for kernel source.\n");
        fclose(fp);
        exit(1);
    }
    size_t read_size = fread(kernel_source, 1, kernel_size, fp);
    if (read_size != kernel_size) {
        fprintf(stderr, "Failed to read kernel source.\n");
        fclose(fp);
        free(kernel_source);
        exit(1);
    }
    kernel_source[kernel_size] = '\0';
    fclose(fp);

    // List platforms
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found. Error code: %d\n", err);
        exit(1);
    }

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);

    // List devices for each platform
    for (cl_uint i = 0; i < num_platforms; ++i) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (num_devices == 0) {
            fprintf(stderr, "No devices found for platform %u.\n", i);
            continue;
        }
        cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        // Output platform and device information...
        free(devices);
    }
    free(platforms);

    // Obtém a plataforma e o dispositivo
    err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform ID. Error code: %d\n", err);
        exit(1);
    }
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get device ID. Error code: %d\n", err);
        exit(1);
    }

    // Cria o contexto OpenCL
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context. Error code: %d\n", err);
        exit(1);
    }

    // Cria a fila de comandos usando a função atual
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    command_queue = clCreateCommandQueueWithProperties(context, device_id, properties, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue. Error code: %d\n", err);
        clReleaseContext(context);
        exit(1);
    }

    // Cria o programa a partir do código-fonte
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &kernel_size, &err);
    free(kernel_source); // Liberar memória alocada para o código do kernel
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program with source. Error code: %d\n", err);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        exit(1);
    }

    // Compila o programa
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Check if the build was successful
    if (err != CL_SUCCESS) {
        // Get the size of the build log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the build log
        char *build_log = (char *)malloc(log_size + 1);
        if (build_log == NULL) {
            fprintf(stderr, "Failed to allocate memory for build log\n");
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            exit(1);
        }

        // Get the build log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);

        // Null-terminate the log
        build_log[log_size] = '\0';

        // Print the build log
        fprintf(stderr, "Error in kernel compilation:\n%s\n", build_log);

        // Clean up
        free(build_log);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);

        // Exit or handle the error
        exit(1);
    }

    // Cria o kernel OpenCL
    kernel = clCreateKernel(program, "deti_coins_opencl_kernel_search", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel. Error code: %d\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        exit(1);
    }

    // Cria o buffer de memória
    device_data = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(uint32_t), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer. Error code: %d\n", err);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        exit(1);
    }

    // Prepara os dados no host
    uint32_t host_data[1024];
    host_data[0] = 1u; // A posição 0 armazena o índice da próxima posição livre

    // Copia os dados do host para o dispositivo
    err = clEnqueueWriteBuffer(command_queue, device_data, CL_TRUE, 0, 1024 * sizeof(uint32_t), host_data, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue write buffer. Error code: %d\n", err);
        clReleaseMemObject(device_data);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        exit(1);
    }


    // Loop de busca
    while (stop_request == 0) {
        // Update random_word
        random_word = random_value_to_try_ascii();

        // Update custom words
        next_value_to_try(custom_word_1);
        if (custom_word_1 == 0x7E7E7E7Eu) {
            custom_word_1 = 0x20202020u;
            next_value_to_try(custom_word_2);
        }
        // Set kernel arguments
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_data);
        err |= clSetKernelArg(kernel, 1, sizeof(uint32_t), &random_word);
        err |= clSetKernelArg(kernel, 2, sizeof(uint32_t), &custom_word_1);
        err |= clSetKernelArg(kernel, 3, sizeof(uint32_t), &custom_word_2);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set kernel arguments. Error code: %d\n", err);
            clReleaseMemObject(device_data);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            exit(1);
        }
        // Executa o kernel
        err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to enqueue NDRange kernel. Error code: %d\n", err);
            break;
        }

        // Aguarda a conclusão da execução
        clFinish(command_queue);

        // Copia os resultados de volta para o host
        err = clEnqueueReadBuffer(command_queue, device_data, CL_TRUE, 0, 1024 * sizeof(uint32_t), host_data, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to enqueue read buffer. Error code: %d\n", err);
            break;
        }

        // Process results
        if (host_data[0] > max_idx)
            max_idx = host_data[0];
        for (idx = 1u; idx < host_data[0] && idx <= 1024u - 13u; idx += 13u) {
            if (idx <= 1024u - 13u) {
                // Convert coin data to characters and print
                char coin_str[53]; // 13 * 4 bytes + null terminator
                for (int i = 0; i < 13; i++) {
                    uint32_t word = host_data[idx + i];
                    for (int j = 0; j < 4; j++) {
                        coin_str[i * 4 + j] = (char)((word >> (j * 8)) & 0xFFu);
                    }
                }
                coin_str[52] = '\0'; // Null terminator
                printf("Coin found: %s\n", coin_str);

                save_deti_coin(&host_data[idx]);
                n_coins++;
            } else {
                fprintf(stderr, "deti_coins_opencl_search: wasted DETI coin\n");
            }
        }

        // Reset host_data[0] to 1u
        host_data[0] = 1u;

        // Copy host_data[0] back to the device
        err = clEnqueueWriteBuffer(command_queue, device_data, CL_TRUE, 0,
                                sizeof(uint32_t), host_data, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to reset device data. Error code: %d\n", err);
            break;
        }

        // Update attempt count
        n_attempts += ((uint64_t)global_work_size * 256u);
    }
    STORE_DETI_COINS();
    printf("deti_coins_opencl_search: %lu DETI coin%s found in %lu attempt%s (expected %.2f coins)\n",
        n_coins, (n_coins == 1ul) ? "" : "s",
        n_attempts, (n_attempts == 1ul) ? "" : "s",
        (double)n_attempts / (double)(1ul << 32));

    // Libera recursos
    clReleaseMemObject(device_data);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

#endif
