
typedef struct _histogram
{
	uint32_t R[256];
	uint32_t G[256];
	uint32_t B[256];
}
histogram;

__kernel void calc_histogram(__global const uchar *img, 
                             __global histogram *hist, 
                             uint height, uint width)
{
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    if (a < height && j < width) {
        __local histogram hist_local = {0};

        const uint pixel = 4 * (i * width + j);
        atomic_add(hist_local.R[img[pixel + 2]], 1);
        atomic_add(hist_local.G[img[pixel + 1]], 1);
        atomic_add(hist_local.B[img[pixel + 0]], 1);
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    const uint color = get_local_id(0);
    const uint value = get_local_id(1);

    if (color >= 3) return;
    if (value >= 256) return;

    switch (color) {
    case 0:
        atomic_add(hist->R[value], hist_local.R[value]);
        break;
    case 1:
        atomic_add(hist->G[value], hist_local.G[value]);
        break;
    case 2:
        atomic_add(hist->B[value], hist_local.B[value]);
        break;
    }
}