
#define SIZE ((size_t) 3 * 256)

__kernel void calc_histogram(__global const uchar *img, __global uint hist[3][256], 
                             uint height, uint width)
{
    const uint g_i = get_global_id(0);
    const uint g_j = get_global_id(1);

    const uint l_i = get_local_id(0);
    const uint l_j = get_local_id(1);

    const uint size_0 = min(get_local_size(0), SIZE);
    const uint size_1 = min(get_local_size(1), SIZE);
    const uint size = size_0 * size_1;

    __global uint *hist_lin = hist;

    __local uint hist_local[3][256];
    __local uint *hist_local_lin = hist_local;

    // nastavi lokalne histograme na 0
    #pragma unroll
    for (uint l_off = 0; l_off < SIZE; l_off += size) {
        const uint i = l_off + l_i * size_0 + l_j;
        if (i >= SIZE) break;

        hist_local_lin[i] = 0;
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    if (g_i < height && g_j < width) {
        const uint pixel = 4 * (g_i * width + g_j);
        atomic_add(&hist_local[0][img[pixel + 2]], 1);
        atomic_add(&hist_local[1][img[pixel + 1]], 1);
        atomic_add(&hist_local[2][img[pixel + 0]], 1);
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (uint l_off = 0; l_off < SIZE; l_off += size) {
        const uint i = l_off + l_i * size_0 + l_j;
        if (i >= SIZE) break;

        atomic_add(&hist_lin[i], hist_local_lin[i]);
    }
}