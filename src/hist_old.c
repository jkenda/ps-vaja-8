#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdbool.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include "FreeImage.h"

#define BINS 256

#define MAX_SOURCE_SIZE 16384
#define WORKGROUP_SIZE 64

typedef struct _histogram
{
	uint32_t R[256];
	uint32_t G[256];
	uint32_t B[256];
}
histogram;

uint32_t max(const uint32_t a, const uint32_t b) { return a > b ? a : b; }

void histogramCPU(uint8_t *image, histogram *H, const uint32_t width, const uint32_t height)
{
    // Each color channel is 1 byte long, there are 4 channels BLUE, GREEN, RED and ALPHA
    // The order is BLUE|GREEN|RED|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			H->R[image[(i * width + j) * 4 + 2]]++;
			H->G[image[(i * width + j) * 4 + 1]]++;
			H->B[image[(i * width + j) * 4 + 0]]++;
		}
	}
}

cl_context context;
cl_command_queue command_queue;
cl_program program;
cl_kernel kernel;

void init(const uint32_t width, const uint32_t height)
{
    // branje datoteke
    FILE *fp = fopen("histogram.cl", "r");
    if (!fp) {
        fprintf(stderr, "cannot open kernel source file\n");
        exit(2);
    }

    char source_str[MAX_SOURCE_SIZE];
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

	// Podatki o platformi
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	clGetPlatformIDs(10, platform_id, &ret_num_platforms);
    printf("num platforms: %d\n", ret_num_platforms);

	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
	clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);
    printf("num devices: %d\n", ret_num_devices);

	// Kontekst
	context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, NULL);

	// Ukazna vrsta
	command_queue = clCreateCommandQueue(context, device_id[0], 0, NULL);

	// Priprava programa
	program = clCreateProgramWithSource(context, 1, (const char **) &source_str, NULL, NULL);

	// Prevajanje
	clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

	// Log
	size_t build_log_len;
	char build_log[32];
	clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 32, build_log, &build_log_len);
	printf("%s\n", build_log);
	if(build_log_len > 2)
		exit(3);

	// kernel: priprava objekta
	kernel = clCreateKernel(program, "calc_histogram", NULL);
}

void cleanup()
{
    clFlush(command_queue);
	clFinish(command_queue);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

void histogramGPU(uint8_t *image, histogram *H, const uint32_t width, uint32_t const height)
{
    // Alokacija pomnilnika na napravi
	cl_mem img_mem_obj  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, image, NULL);
	cl_mem hist_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(histogram), NULL, NULL);

	// Delitev dela
	size_t local_item_size[2] = { max(WORKGROUP_SIZE, 3), max(WORKGROUP_SIZE, 256) };
	size_t num_groups[2] = { (height - 1) / local_item_size[0] + 1 , (width - 1) / local_item_size[1] + 1 };
	size_t global_item_size[2] = { num_groups[0] * local_item_size[0], num_groups[1] * local_item_size[1] };

	// kernel: argumenti
	clSetKernelArg(kernel, 0, sizeof(cl_mem),  (void *) &img_mem_obj);
	clSetKernelArg(kernel, 1, sizeof(cl_mem),  (void *) &hist_mem_obj);
	clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *) &height);
	clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *) &width);

	// kernel: zagon
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

	// Kopiranje rezultatov
	clEnqueueReadBuffer(command_queue, hist_mem_obj, CL_TRUE, 0, sizeof(histogram), H, 0, NULL, NULL);

	clReleaseMemObject(img_mem_obj);
	clReleaseMemObject(hist_mem_obj);
}

void printHistogram(const histogram *H) {
	printf("Colour\tNo. Pixels\n");
	for (int i = 0; i < BINS; i++) {
		if (H->B[i]>0)
			printf("%dB\t%d\n", i, H->B[i]);
		if (H->G[i]>0)
			printf("%dG\t%d\n", i, H->G[i]);
		if (H->R[i]>0)
			printf("%dR\t%d\n", i, H->R[i]);
	}
}

bool equal(const histogram *A, const histogram *B)
{
	for (int i = 0; i < BINS; i++) {
		if (A->R[i] != B->R[i]) return false;
		if (A->G[i] != B->G[i]) return false;
		if (A->B[i] != B->B[i]) return false;
	}
	return true;
}

int main(int argc, char **argv)
{
	if (argc != 2) return 1;

	const char *filename = argv[1];

    // Load image from file
	FIBITMAP *imageBitmap = FreeImage_Load(FIF_BMP, filename, 0);
	// Convert it to a 32-bit image
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    // Get image dimensions
    uint32_t width  = FreeImage_GetWidth(imageBitmap32);
	uint32_t height = FreeImage_GetHeight(imageBitmap32);
	uint32_t pitch  = FreeImage_GetPitch(imageBitmap32);
	// Preapare room for a raw data copy of the image
    uint8_t *image = (uint8_t *) malloc(height * pitch * sizeof(uint8_t));

    // Extract raw data from the image
	FreeImage_ConvertToRawBits(image, imageBitmap, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    // Free source image data
	FreeImage_Unload(imageBitmap32);
	FreeImage_Unload(imageBitmap);

    init(width, height);

    // Compute and print the histogram
    histogram A = {0}, B = {0};
	histogramCPU(image, &A, width, height);
	histogramGPU(image, &B, width, height);
    
	printHistogram(&B);
	printf("%s\n", equal(&A, &B) ? "equal" : "not equal");

    cleanup();

	return 0;
}