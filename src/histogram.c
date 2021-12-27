#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdbool.h>
#include <CL/cl.h>
#include <time.h>
#include "FreeImage.h"
#include "cl_errors.c"

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
		perf_t perf_640_480gpu   = cas_izvajanja("test/640x480.jpg",   wgsize, 10, 20);
		printf("%12lf ", perf_640_480gpu.t_gpu); fflush(stdout);
		perf_t perf_800_600gpu   = cas_izvajanja("test/800x600.jpg",   wgsize, 10, 20);
		printf("%12lf ", perf_800_600gpu.t_gpu); fflush(stdout);
		perf_t perf_1600_900gpu  = cas_izvajanja("test/1600x900.jpg",  wgsize, 10, 20);
		printf("%12lf ", perf_1600_900gpu.t_gpu); fflush(stdout);
		perf_t perf_1920_1080gpu = cas_izvajanja("test/1920x1080.jpg", wgsize, 10, 20);
		printf("%12lf ", perf_1920_1080gpu.t_gpu); fflush(stdout);
		perf_t perf_3840_2160gpu = cas_izvajanja("test/3840x2160.jpg", wgsize, 10, 20);
		printf("%12lf ", perf_3840_2160gpu.t_gpu); fflush(stdout);
    	perf_t perf_8000_8000gpu = cas_izvajanja("test/8000x8000.jpg", wgsize, 10, 20);
		printf("%12lf ", perf_8000_8000gpu.t_gpu); fflush(stdout);

        printf("%.3lf,%.3lf,%.3lf,%.3lf,%.3lf,%.3lf\n",
			perf_640_480gpu.speedup, perf_800_600gpu.speedup, perf_1600_900gpu.speedup, 
			perf_1920_1080gpu.speedup, perf_3840_2160gpu.speedup, perf_8000_8000gpu.speedup
        );
    }

	cl_finalize();

	return 0;
}