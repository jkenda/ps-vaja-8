
__kernel void calc_histogram(__global const uchar *img, __global uint hist[3][256], 
                             uint height, uint width)
{
    const uint g_i = get_global_id(0);
    const uint g_j = get_global_id(1);

    const uint l_i = get_local_id(0);
    const uint l_j = get_local_id(1);

    const uint size_0 = get_local_size(0);
    const uint size_1 = get_local_size(1);
    const uint size = size_0 * size_1;

    __local uint hist_local[3][256];

    // nastavi lokalne histograme na 0
    if (size_0 < 16) {
        for (uint l_off = 0; l_off < 256; l_off += size) {
            const uint i = l_off + l_i * size_0 + l_j;
            if (i >= 256) return;
            hist_local[0][i] = 0;
            hist_local[1][i] = 0;
            hist_local[2][i] = 0;
        }
    }
    else if (l_i < 16 && l_j < 16) {
        const uint i = l_i * 16 + l_j;
        if (i >= 256) return;
        hist_local[0][i] = 0;
        hist_local[1][i] = 0;
        hist_local[2][i] = 0;
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    if (g_i < height && g_j < width) {
        const uint pixel = 4 * (g_i * width + g_j);
        atomic_add(&hist_local[0][img[pixel + 2]], 1);
        atomic_add(&hist_local[1][img[pixel + 1]], 1);
        atomic_add(&hist_local[2][img[pixel + 0]], 1);
    }

    if (l_i >= 16 || l_j >= 16) return;

	barrier(CLK_LOCAL_MEM_FENCE);

    if (size_0 < 16) {
        for (uint l_off = 0; l_off < 256; l_off += size_0 * size_1) {
            const uint i = l_off + l_i * size_0 + l_j;
            if (i >= 256) return;

            atomic_add(&hist[0][i], hist_local[0][i]);
            atomic_add(&hist[1][i], hist_local[1][i]);
            atomic_add(&hist[2][i], hist_local[2][i]);
        }
    }
    else {
        const uint i = l_i * 16 + l_j;
        if (i >= 256) return;
        atomic_add(&hist[0][i], hist_local[0][i]);
        atomic_add(&hist[1][i], hist_local[1][i]);
        atomic_add(&hist[2][i], hist_local[2][i]);
    }
}