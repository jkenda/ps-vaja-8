#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdbool.h>
#include <CL/cl.h>
#include <time.h>
#include "FreeImage.h"

#define BINS 256
#define MAX_SOURCE_SIZE 16384

typedef struct 
{
	uint32_t R[256];
	uint32_t G[256];
	uint32_t B[256];
}
histogram_t;

const uint32_t zero = 0U;

typedef struct
{
	double t_cpu, t_gpu, speedup;
}
perf_t;


cl_context context;
cl_program program;
cl_command_queue command_queue;
cl_kernel kernel;
cl_mem hist_mem_obj;

const char *errors[] = {
    "CL_SUCCESS"                                      ,
    "CL_DEVICE_NOT_FOUND"                             ,
    "CL_DEVICE_NOT_AVAILABLE"                         ,
    "CL_COMPILER_NOT_AVAILABLE"                       ,
    "CL_MEM_OBJECT_ALLOCATION_FAILURE"                ,
    "CL_OUT_OF_RESOURCES"                             ,
    "CL_OUT_OF_HOST_MEMORY"                           ,
    "CL_PROFILING_INFO_NOT_AVAILABLE"                 ,
    "CL_MEM_COPY_OVERLAP"                             ,
    "CL_IMAGE_FORMAT_MISMATCH"                        ,
    "CL_IMAGE_FORMAT_NOT_SUPPORTED"                   ,
    "CL_BUILD_PROGRAM_FAILURE"                        ,
    "CL_MAP_FAILURE"                                  ,
    "CL_MISALIGNED_SUB_BUFFER_OFFSET"                 ,
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"    ,
    "CL_COMPILE_PROGRAM_FAILURE"                      ,
    "CL_LINKER_NOT_AVAILABLE"                         ,
    "CL_LINK_PROGRAM_FAILURE"                         ,
    "CL_DEVICE_PARTITION_FAILED"                      ,
    "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"                ,
    "CL_INVALID_VALUE"                                ,
    "CL_INVALID_DEVICE_TYPE"                          ,
    "CL_INVALID_PLATFORM"                             ,
    "CL_INVALID_DEVICE"                               ,
    "CL_INVALID_CONTEXT"                              ,
    "CL_INVALID_QUEUE_PROPERTIES"                     ,
    "CL_INVALID_COMMAND_QUEUE"                        ,
    "CL_INVALID_HOST_PTR"                             ,
    "CL_INVALID_MEM_OBJECT"                           ,
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"              ,
    "CL_INVALID_IMAGE_SIZE"                           ,
    "CL_INVALID_SAMPLER"                              ,
    "CL_INVALID_BINARY"                               ,
    "CL_INVALID_BUILD_OPTIONS"                        ,
    "CL_INVALID_PROGRAM"                              ,
    "CL_INVALID_PROGRAM_EXECUTABLE"                   ,
    "CL_INVALID_KERNEL_NAME"                          ,
    "CL_INVALID_KERNEL_DEFINITION"                    ,
    "CL_INVALID_KERNEL"                               ,
    "CL_INVALID_ARG_INDEX"                            ,
    "CL_INVALID_ARG_VALUE"                            ,
    "CL_INVALID_ARG_SIZE"                             ,
    "CL_INVALID_KERNEL_ARGS"                          ,
    "CL_INVALID_WORK_DIMENSION"                       ,
    "CL_INVALID_WORK_GROUP_SIZE"                      ,
    "CL_INVALID_WORK_ITEM_SIZE"                       ,
    "CL_INVALID_GLOBAL_OFFSET"                        ,
    "CL_INVALID_EVENT_WAIT_LIST"                      ,
    "CL_INVALID_EVENT"                                ,
    "CL_INVALID_OPERATION"                            ,
    "CL_INVALID_GL_OBJECT"                            ,
    "CL_INVALID_BUFFER_SIZE"                          ,
    "CL_INVALID_MIP_LEVEL"                            ,
    "CL_INVALID_GLOBAL_WORK_SIZE"                     ,
    "CL_INVALID_PROPERTY"                             ,
    "CL_INVALID_IMAGE_DESCRIPTOR"                     ,
    "CL_INVALID_COMPILER_OPTIONS"                     ,
    "CL_INVALID_LINKER_OPTIONS"                       ,
    "CL_INVALID_DEVICE_PARTITION_COUNT"               ,
};

const char *cl_error(int status)
{
    return errors[-status];
}

uint32_t max(const uint32_t a, const uint32_t b) { return a >= b ? a : b; }

void cl_init()
{
	cl_int status;

    // branje datoteke
    FILE *fp = fopen("src/histogram.cl", "r");
    if(!fp)
    {
        fprintf(stderr, "cannot open kernel file\n");
        exit(2);
    }

	// preberi kernel file
    char *source_str = malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

	// Podatki o platformi
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	status = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
	printf("ids: %s\n", cl_error(status));

	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
	status = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);
	printf("devices: %s\n", cl_error(status));

	// Kontekst
	context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, NULL);

	// Ukazna vrsta
	command_queue = clCreateCommandQueue(context, device_id[0], 0, NULL);

	// Priprava programa
	program = clCreateProgramWithSource(context, 1, (const char **) &source_str, NULL, NULL);

	// Prevajanje
	status = clBuildProgram(program, 1, device_id, NULL, NULL, NULL);
	printf("build: %s\n", cl_error(status));

	if (status != 0) {
		// Log
		size_t build_log_len;
		char *build_log;
		status = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);

		build_log = (char *) malloc(build_log_len + 1);
		clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
		printf("%s\n", build_log);
		free(build_log);
		if(build_log_len > 2)	
			exit(3);
	}

	// kernel: priprava objekta
	kernel = clCreateKernel(program, "calc_histogram", NULL);

	hist_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(histogram_t), NULL, &status);
	printf("make buffer: %s\n", cl_error(status));

	free(source_str);
}

void cl_finalize()
{
	clReleaseMemObject(hist_mem_obj);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

void histogramCPU(histogram_t *H, uint8_t *image, uint32_t width, uint32_t height, uint32_t wgsize)
{
	memset(H, 0, sizeof(histogram_t));
    // Each color channel is 1 byte long, there are 4 channels BLUE, GREEN, RED and ALPHA
    // The order is BLUE|GREEN|RED|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
		{
			H->R[image[(i * width + j) * 4 + 2]]++;
			H->G[image[(i * width + j) * 4 + 1]]++;
			H->B[image[(i * width + j) * 4 + 0]]++;
		}
	}
}

void histogramGPU(histogram_t *H, uint8_t *image, uint32_t width, uint32_t height, uint32_t wgsize)
{
	cl_int status;

	// Delitev dela
	size_t local_item_size[] = { wgsize, wgsize };
	size_t num_groups[] = { (max(height, 3) - 1) / local_item_size[0] + 1 , (max(width, 256) - 1) / local_item_size[1] + 1 };
	size_t global_item_size[] = { num_groups[0] * local_item_size[0], num_groups[1] * local_item_size[1] };
	//printf("global_item_size (%u, %u)\n", global_item_size[0], global_item_size[1]);

	// Alokacija pomnilnika na napravi
	cl_mem img_mem_obj  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, image, &status);
	//printf("make buffer: %s\n", cl_error(status));

	// kernel: argumenti
	status  = clSetKernelArg(kernel, 0, sizeof(cl_mem),  (void *) &img_mem_obj);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem),  (void *) &hist_mem_obj);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *) &height);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *) &width);
	//printf("arg: %s\n", cl_error(status));

	status = clEnqueueFillBuffer(command_queue, hist_mem_obj, &zero, sizeof(uint32_t), 0, sizeof(histogram_t), 0, NULL, NULL);
	// printf("fill: %s\n", cl_error(status)); fflush(stdout);

	// kernel: zagon
	status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
	// printf("enqueue: %s\n", cl_error(status));

	// Kopiranje rezultatov
	status = clEnqueueReadBuffer(command_queue, hist_mem_obj, CL_TRUE, 0, sizeof(histogram_t), H, 0, NULL, NULL);
	//printf("read: %s\n", cl_error(status));

	// čiščenje
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseMemObject(img_mem_obj);
}

void printHistogram(histogram_t *H) {
	printf("Colour\tNo. Pixels\n");
	for (int i = 0; i < BINS; i++) {
		if (H->B[i] > 0)
			printf("%dB\t%d\n", i, H->B[i]);
		if (H->G[i] > 0)
			printf("%dG\t%d\n", i, H->G[i]);
		if (H->R[i] > 0)
			printf("%dR\t%d\n", i, H->R[i]);
	}
}

bool equal(histogram_t *A, histogram_t *B)
{
	for (int i = 0; i < BINS; i++) {
		if (A->R[i] != B->R[i]) return false;
		if (A->G[i] != B->G[i]) return false;
		if (A->B[i] != B->B[i]) return false;
	}
	return true;
}

perf_t cas_izvajanja(const char *filename, const uint32_t wgsize, const uint32_t samples_cpu, const uint32_t samples_gpu)
{
    struct timespec start, finish;
    perf_t perf;

    // Load image from file
	FIBITMAP *imageJpeg = FreeImage_Load(FIF_JPEG, filename, 0);
	// Convert it to a 32-bit image
    FIBITMAP *imageJpeg32 = FreeImage_ConvertTo32Bits(imageJpeg);

    // Get image dimensions
    uint32_t width  = FreeImage_GetWidth(imageJpeg32);
	uint32_t height = FreeImage_GetHeight(imageJpeg32);
	uint32_t pitch  = FreeImage_GetPitch(imageJpeg32);
	// Preapare room for a raw data copy of the image
    uint8_t *image = (uint8_t *) malloc(height * pitch * sizeof(uint8_t));

    // Extract raw data from the image
	FreeImage_ConvertToRawBits(image, imageJpeg, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    // Free source image data
	FreeImage_Unload(imageJpeg32);
	FreeImage_Unload(imageJpeg);

    // Compute and print the histogram
	histogram_t A, B;

    clock_gettime(CLOCK_MONOTONIC, &start);
	for (int i = 0; i < samples_cpu; i++) {
		histogramCPU(&A, image, width, height, 0);
	}
    clock_gettime(CLOCK_MONOTONIC, &finish);

	perf.t_cpu = (finish.tv_sec - start.tv_sec);
    perf.t_cpu += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	perf.t_cpu /= samples_cpu;

    clock_gettime(CLOCK_MONOTONIC, &start);
	for (int i = 0; i < samples_gpu; i++) {
		memset(&B, 0, sizeof(histogram_t));
    	histogramGPU(&B, image, width, height, wgsize);
	}
    clock_gettime(CLOCK_MONOTONIC, &finish);

    perf.t_gpu = (finish.tv_sec - start.tv_sec);
    perf.t_gpu += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	perf.t_gpu /= samples_gpu;

	perf.speedup = perf.t_cpu / perf.t_gpu;

	free(image);

    return equal(&A, &B) ? perf : (perf_t) { 0, 0, 0 };
}

int main(int argc, const char **argv)
{
	cl_init();

    printf("%7s %12s %12s %12s %12s %12s %12s %s\n",
		"WG size", "640x480", "800x600", "1600x900", "1920x1080", "3840x2160", "8000x8000", "pohitritev");
	fflush(stdout);

    for (int wgsize = 4; wgsize <= 32; wgsize *= 2) {
		printf("%7u ", wgsize); fflush(stdout);
		perf_t perf_640_480gpu   = cas_izvajanja("test/640x480.jpg",   wgsize, 1, 2);
		printf("%12lf ", perf_640_480gpu.t_gpu); fflush(stdout);
		perf_t perf_800_600gpu   = cas_izvajanja("test/800x600.jpg",   wgsize, 1, 2);
		printf("%12lf ", perf_800_600gpu.t_gpu); fflush(stdout);
		perf_t perf_1600_900gpu  = cas_izvajanja("test/1600x900.jpg",  wgsize, 1, 2);
		printf("%12lf ", perf_1600_900gpu.t_gpu); fflush(stdout);
		perf_t perf_1920_1080gpu = cas_izvajanja("test/1920x1080.jpg", wgsize, 1, 2);
		printf("%12lf ", perf_1920_1080gpu.t_gpu); fflush(stdout);
		perf_t perf_3840_2160gpu = cas_izvajanja("test/3840x2160.jpg", wgsize, 1, 2);
		printf("%12lf ", perf_3840_2160gpu.t_gpu); fflush(stdout);
    	perf_t perf_8000_8000gpu = cas_izvajanja("test/8000x8000.jpg", wgsize, 1, 2);
		printf("%12lf ", perf_8000_8000gpu.t_gpu); fflush(stdout);

        printf("%.3lf,%.3lf,%.3lf,%.3lf,%.3lf,%.3lf\n",
			perf_640_480gpu.speedup, perf_800_600gpu.speedup, perf_1600_900gpu.speedup, 
			perf_1920_1080gpu.speedup, perf_3840_2160gpu.speedup, perf_8000_8000gpu.speedup
        );
    }

	cl_finalize();

	return 0;
}

/*
ids: CL_SUCCESS
devices: CL_SUCCESS
build: CL_SUCCESS
make buffer: CL_SUCCESS
WG size      640x480      800x600     1600x900    1920x1080    3840x2160    8000x8000 pohitritev
      4     0.001867     0.002323     0.006757     0.010516     0.041879     0.565632 1.398,1.751,1.805,1.672,1.699,1.518
      8     0.015547     0.001357     0.003179     0.005027     0.019031     0.348019 0.167,3.000,3.836,3.504,3.737,2.465
     16     0.015564     0.001321     0.003053     0.004667     0.017340     0.353270 0.167,3.079,3.994,3.774,4.101,2.428
     32     0.015677     0.001569     0.003977     0.006138     0.023638     0.539223 0.166,2.599,3.067,2.869,3.010,1.591

*/