#ifndef DETI_COINS_OPENCL_SEARCH
#define DETI_COINS_OPENCL_SEARCH

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "search_utilities.h"
#include <time.h>

// Define u32_t, u08_t, and u64_t
typedef unsigned int u32_t;
typedef unsigned char u08_t;
typedef unsigned long long u64_t;

// Gera um valor aleatório de 32 bits com caracteres ASCII imprimíveis
static u32_t random_value_to_try_ascii() {
    u32_t value = 0;
    for (int i = 0; i < 4; i++) {
        u08_t random_char = (u08_t)(0x20 + (rand() % (0x7F - 0x20)));
        value |= ((u32_t)random_char) << (i * 8);
    }
    return value;
}

static void deti_coins_opencl_search(u32_t n_random_words)
{
    u32_t idx, max_idx, random_word, custom_word_1, custom_word_2;
    u64_t n_attempts = 0ul, n_coins = 0ul;
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

    // Carrega o código do kernel OpenCL
    const char *kernel_source = 
    "__kernel void deti_coins_opencl_kernel_search("
    "   __global uint *deti_coins_storage_area,"
    "   uint random_word,"
    "   uint custom_word_1,"
    "   uint custom_word_2) {"
    "   // Implementação do kernel aqui..."
    "}";

    // Obtém a plataforma e o dispositivo
    err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Cria o contexto OpenCL
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

    // Cria a fila de comandos
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Cria o programa a partir do código-fonte
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);

    // Compila o programa
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Cria o kernel OpenCL
    kernel = clCreateKernel(program, "deti_coins_opencl_kernel_search", &err);

    // Cria o buffer de memória
    device_data = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(u32_t), NULL, &err);

    // Prepara os dados no host
    u32_t host_data[1024];
    host_data[0] = 1u; // A posição 0 armazena o índice da próxima posição livre

    // Copia os dados do host para o dispositivo
    err = clEnqueueWriteBuffer(command_queue, device_data, CL_TRUE, 0, 1024 * sizeof(u32_t), host_data, 0, NULL, NULL);

    // Define os argumentos do kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&device_data);
    err |= clSetKernelArg(kernel, 1, sizeof(u32_t), (void *)&random_word);
    err |= clSetKernelArg(kernel, 2, sizeof(u32_t), (void *)&custom_word_1);
    err |= clSetKernelArg(kernel, 3, sizeof(u32_t), (void *)&custom_word_2);

    // Loop de busca
    while (stop_request == 0) {
        // Executa o kernel
        err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

        // Aguarda a conclusão da execução
        clFinish(command_queue);

        // Copia os resultados de volta para o host
        err = clEnqueueReadBuffer(command_queue, device_data, CL_TRUE, 0, 1024 * sizeof(u32_t), host_data, 0, NULL, NULL);

        // Processa os resultados
        if (host_data[0] > max_idx)
            max_idx = host_data[0];
        for (idx = 1u; idx < host_data[0] && idx <= 1024 - 13u; idx += 13u) {
            if (idx <= 1024 - 13u) {
                printf("host_data[%u] = %08X\n", idx, host_data[idx]);
                save_deti_coin(&host_data[idx]);
                n_coins++;
            } else {
                fprintf(stderr, "deti_coins_opencl_search: wasted DETI coin\n");
            }
        }
        if (custom_word_1 != 0x7E7E7E7Eu) {
            next_value_to_try(custom_word_1); 
        } else {
            custom_word_1 = 0x20202020u;
            next_value_to_try(custom_word_2);
        }

        n_attempts += (64ul << 20);
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
