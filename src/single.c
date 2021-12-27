#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "FreeImage.h"

#define BINS 256

typedef struct 
{
	uint32_t R[256];
	uint32_t G[256];
	uint32_t B[256];
}
histogram;

void histogramCPU(uint8_t *imageIn, histogram *H, int width, int height)
{
    //Each color channel is 1 byte long, there are 4 channels BLUE, GREEN, RED and ALPHA
    //The order is BLUE|GREEN|RED|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
	for (int i = 0; i < (height); i++) {
		for (int j = 0; j < (width); j++)
		{
			H->B[imageIn[(i * width + j) * 4 + 0]]++;
			H->G[imageIn[(i * width + j) * 4 + 1]]++;
			H->R[imageIn[(i * width + j) * 4 + 2]]++;
		}
	}
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
int main(int argc, const char **argv)
{
	if (argc != 2) return 1;

	const char *filename = argv[1];

    // Load image from file
	FIBITMAP *imageBitmap = FreeImage_Load(FIF_BMP, filename, 0);
	// Convert it to a 32-bit image
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    // Get image dimensions
    int width  = FreeImage_GetWidth(imageBitmap32);
	int height = FreeImage_GetHeight(imageBitmap32);
	int pitch  = FreeImage_GetPitch(imageBitmap32);
	//Preapare room for a raw data copy of the image
    unsigned char *imageIn = (uint8_t *) malloc(height*pitch * sizeof(uint8_t));

    // Initalize the histogram
    histogram H = {0};

    // Extract raw data from the image
	FreeImage_ConvertToRawBits(imageIn, imageBitmap, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    // Free source image data
	FreeImage_Unload(imageBitmap32);
	FreeImage_Unload(imageBitmap);

    // Compute and print the histogram
	histogramCPU(imageIn, &H, width, height);
	printHistogram(&H);

	return 0;
}