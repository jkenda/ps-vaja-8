
__kernel void calc_histogram(__global const uchar *img, __global uint hist[3][256], 
                             uint height, uint width)
{
    const uint g_i = get_global_id(0);
    const uint g_j = get_global_id(1);

    const uint l_i = get_local_id(0);
    const uint l_j = get_local_id(1);

    const uint l_size = get_local_size(1);

    __local uint hist_local[3][256];

    // nastavi lokalne histograme na 0
    if (l_i < 3) {
        #pragma unroll
        for (uint j = l_j; j < 256; j += l_size) {
            hist_local[l_i][j] = 0;
        }
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    if (g_i < height && g_j < width) {
        const uint pixel = 4 * (g_i * width + g_j);
        atomic_add(&hist_local[0][img[pixel + 2]], 1);
        atomic_add(&hist_local[1][img[pixel + 1]], 1);
        atomic_add(&hist_local[2][img[pixel + 0]], 1);
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i < 3) {
        #pragma unroll
        for (uint j = l_j; j < 256; j += l_size) {
            atomic_add(&hist[l_i][j], hist_local[l_i][j]);
        }
    }
}