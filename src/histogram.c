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

uint32_t workgroup_size;

typedef struct 
{
	uint32_t R[256];
	uint32_t G[256];
	uint32_t B[256];
}
histogram;

void init()
{
	
}

histogram histogramCPU(uint8_t *image, uint32_t width, uint32_t height)
{
    // Initalize the histogram
    histogram H = {0};

    // Each color channel is 1 byte long, there are 4 channels BLUE, GREEN, RED and ALPHA
    // The order is BLUE|GREEN|RED|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
	for (int i = 0; i < (height); i++) {
		for (int j = 0; j < (width); j++)
		{
			H.R[image[(i * width + j) * 4 + 2]]++;
			H.G[image[(i * width + j) * 4 + 1]]++;
			H.B[image[(i * width + j) * 4 + 0]]++;
		}
	}

    return H;
}

histogram histogramGPU(uint8_t *image, uint32_t width, uint32_t height)
{
    char *source_str;
    size_t source_size;
	cl_int status;

    // branje datoteke
    FILE *fp = fopen("src/histogram.cl", "r");
    if(!fp)
    {
        fprintf(stderr, "cannot open kernel file\n");
        exit(2);
    }

	// preberi kernel file
    source_str = (char*) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
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
	cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, NULL);

	// Ukazna vrsta
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, NULL);

	// Delitev dela
	size_t local_item_size = workgroup_size;
	size_t num_groups = (width * height - 1) / local_item_size + 1;
	size_t global_item_size = num_groups * local_item_size;

	// Alokacija pomnilnika na napravi
	cl_mem img_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4, image, NULL);
	cl_mem hist_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(histogram), NULL, NULL);

	// Priprava programa
	cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str, NULL, NULL);

	// Prevajanje
	status = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
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
	cl_kernel kernel = clCreateKernel(program, "calc_histogram", NULL);

	// kernel: argumenti
	status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &img_mem_obj);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &hist_mem_obj);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *) &height);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *) &width);
	printf("arg: %s\n", cl_error(status));

	// kernel: zagon
	status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	printf("enqueue: %s\n", cl_error(status));

    // Initalize the histogram
    histogram H;

	// Kopiranje rezultatov
	status = clEnqueueReadBuffer(command_queue, hist_mem_obj, CL_TRUE, 0, sizeof(histogram), &H, 0, NULL, NULL);
	printf("read: %s\n", cl_error(status));

	// čiščenje
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(img_mem_obj);
	clReleaseMemObject(hist_mem_obj);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
    free(source_str);

    return H;
}

void printHistogram(histogram *H) {
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

bool equal(histogram *A, histogram *B)
{
	for (int i = 0; i < BINS; i++) {
		if (A->R[i] != B->R[i]) return false;
		if (A->G[i] != B->G[i]) return false;
		if (A->B[i] != B->B[i]) return false;
	}
	return true;
}

double cas_izvajanja(const histogram (*fun)(uint8_t *, uint32_t, uint32_t), 
					 const char *filename, const uint32_t wgsize, const uint32_t samples)
{
    struct timespec start, finish;
    double elapsed;

	workgroup_size = wgsize;

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
	histogram A = histogramCPU(image, width, height);
	histogram B;

    clock_gettime(CLOCK_MONOTONIC, &start);

	for (int i = 0; i < samples; i++) {
    	B = fun(image, width, height);
	}

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	
	//return elapsed / samples;
    return equal(&A, &B) ? elapsed / samples : -1;
}

int main(int argc, const char **argv)
{
    double time_640_480cpu   = cas_izvajanja(histogramCPU, "test/640x480.jpg",   0, 100);
    double time_800_600cpu   = cas_izvajanja(histogramCPU, "test/800x600.jpg",   0, 100);
    double time_1600_900cpu  = cas_izvajanja(histogramCPU, "test/1600x900.jpg",  0, 100);
    double time_1920_1080cpu = cas_izvajanja(histogramCPU, "test/1920x1080.jpg", 0, 100);
    double time_3840_2160cpu = cas_izvajanja(histogramCPU, "test/3840x2160.jpg", 0, 100);
    double time_8000_8000cpu = cas_izvajanja(histogramCPU, "test/8000x8000.jpg", 0, 100);

	puts("sekvenčno");
    printf("%12s %12s %12s %12s %12s %12s\n",
		"640x480", "800x600", "1600x900", "1920x1080", "3840x2160", "8000x8000");
	printf("%12lf %12lf %12lf %12lf %12lf %12lf\n\n",
		time_640_480cpu, time_800_600cpu, time_1600_900cpu, time_1920_1080cpu, time_3840_2160cpu, time_8000_8000cpu
	);

	puts("paralelno");
    printf("%7s %12s %12s %12s %12s %12s %12s %s\n",
		"WG_size", "640x480", "800x600", "1600x900", "1920x1080", "3840x2160", "8000x8000", "pohitritev");
	
    for (int wgsize = 16; wgsize <= 512; wgsize *= 2) {
		double time_640_480gpu   = cas_izvajanja(histogramGPU, "test/640x480.jpg",   wgsize, 1000);
		double time_800_600gpu   = cas_izvajanja(histogramGPU, "test/800x600.jpg",   wgsize, 1000);
		double time_1600_900gpu  = cas_izvajanja(histogramGPU, "test/1600x900.jpg",  wgsize, 1000);
		double time_1920_1080gpu = cas_izvajanja(histogramGPU, "test/1920x1080.jpg", wgsize, 1000);
		double time_3840_2160gpu = cas_izvajanja(histogramGPU, "test/3840x2160.jpg", wgsize, 1000);
    	double time_8000_8000gpu = cas_izvajanja(histogramGPU, "test/8000x8000.jpg", wgsize, 1000);

		double pohitritev_640_480   = time_640_480cpu   / time_640_480gpu   ;
		double pohitritev_800_600   = time_800_600cpu   / time_800_600gpu   ;
		double pohitritev_1600_900  = time_1600_900cpu  / time_1600_900gpu  ;
		double pohitritev_1920_1080 = time_1920_1080cpu / time_1920_1080gpu ;
		double pohitritev_3840_2160 = time_3840_2160cpu / time_3840_2160gpu ;
		double pohitritev_8000_8000 = time_8000_8000cpu / time_8000_8000gpu ;

        printf("%7d %12lf %12lf %12lf %12lf %12lf %12lf %.3lf,%.3lf,%.3lf,%.3lf,%.3lf,%.3lf\n",
            wgsize, time_640_480gpu, time_800_600gpu, time_1600_900gpu, time_1920_1080gpu, time_3840_2160gpu, time_8000_8000gpu,
			pohitritev_640_480, pohitritev_800_600, pohitritev_1600_900, pohitritev_1920_1080, pohitritev_3840_2160, pohitritev_8000_8000
        );
    }

	return 0;
}