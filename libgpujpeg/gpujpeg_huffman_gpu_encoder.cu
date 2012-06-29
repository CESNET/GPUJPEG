/**
 * Copyright (c) 2011, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
 
#include "gpujpeg_huffman_gpu_encoder.h"
#include "gpujpeg_util.h"

#define WARPS_NUM 8


#ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
/** Allocate huffman tables in constant memory */
__constant__ struct gpujpeg_table_huffman_encoder gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
/** Pass huffman tables to encoder */
extern struct gpujpeg_table_huffman_encoder (*gpujpeg_encoder_table_huffman)[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT] = &gpujpeg_huffman_gpu_encoder_table_huffman;
#endif

/** Natural order in constant memory */
__constant__ int gpujpeg_huffman_gpu_encoder_order_natural[GPUJPEG_ORDER_NATURAL_SIZE];

/**
 * Write marker to compressed data
 * 
 * @param data_compressed  Data compressed
 * @oaran marker  Marker to write (JPEG_MARKER_...)
 * @return void
 */
#define gpujpeg_huffman_gpu_encoder_marker(data_compressed, marker) { \
    *data_compressed = 0xFF;\
    data_compressed++; \
    *data_compressed = (uint8_t)(marker); \
    data_compressed++; }


/**
 * Adds up to 24 bits at once.
 */
__device__ inline void 
gpujpeg_huffman_gpu_encoder_emit_bits(int & buffer_bits, uint8_t * const buffer_ptr, int code_bit_size, unsigned int code_word) {
    while ( code_bit_size ) {
        // get pointer to current output byte and number of remianing bits in it 
        uint8_t * const out_byte_ptr = buffer_ptr + (buffer_bits >> 3);
        const int old_bit_count = buffer_bits & 7;
        const int new_bit_count = min(8 - old_bit_count, code_bit_size);
        
        const uint8_t out_byte = ((old_bit_count ? *out_byte_ptr : 0) << new_bit_count)
                               | (code_word >> (code_bit_size - new_bit_count) & ((1 << new_bit_count) - 1));
        
        out_byte_ptr[0] = out_byte;
        code_bit_size -= new_bit_count;
        buffer_bits += new_bit_count;
    }
}


__device__ static void
gpujpeg_huffman_gpu_encode_value(int & out_nbits, int & out_cword, const int preceding_zero_count, const int value,
                                 const struct gpujpeg_table_huffman_encoder * const d_table) {
    out_cword = value;
    int absolute = value;
    if ( value < 0 ) {
        // valu eis now absolute value of input
        absolute = -absolute;
        // For a negative input, want temp2 = bitwise complement of abs(input)
        // This code assumes we are on a two's complement machine
        out_cword--;
    }

    // Find the number of bits needed for the magnitude of the coefficient
    out_nbits = 0;
    while ( absolute ) {
        out_nbits++;
        absolute >>= 1;
    }
    
    // trim remaining bits
    out_cword &= (1 << out_nbits) - 1;
    
    // find prefix of the codeword and size of the prefix
    const int prefix_idx = preceding_zero_count * 16 + out_nbits;
    out_cword |= d_table->code[prefix_idx] << out_nbits;
    out_nbits += d_table->size[prefix_idx];
}


/**
 * Encode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ int
gpujpeg_huffman_gpu_encoder_encode_block(const int value0, const int leftover_value, int & out_bits, int & dc, int * data, uint8_t* & data_compressed, 
    struct gpujpeg_table_huffman_encoder* d_table_dc, struct gpujpeg_table_huffman_encoder* d_table_ac)
{
    // Encode the DC coefficient difference per section F.1.2.1
    int temp = value0 - dc;
    dc = value0;
    
    __threadfence_block();
    // put leftover bits back
    *data = leftover_value;
    
    int cword_size, cword_value;
    gpujpeg_huffman_gpu_encode_value(cword_size, cword_value, 0, temp, d_table_dc);
    gpujpeg_huffman_gpu_encoder_emit_bits(out_bits, (uint8_t*)data, cword_size, cword_value);
    
    // Encode the AC coefficients per section F.1.2.2 (r = run length of zeros)
    for ( int k = 1; k < 64; k++ ) 
    {
        __threadfence_block();
        const int size = data[k] & 31;
        const int value = data[k] >> 5;
        __threadfence_block();
        gpujpeg_huffman_gpu_encoder_emit_bits(out_bits, (uint8_t*)data, size, value);
    }

    return 0;
}



// apply byte stuffing to encoded bytes
__device__ void
gpujpeg_huffman_gpu_encoder_byte_stuff(uint8_t* const src, int & remaining_bits, uint8_t * const dest, int & remaining_bytes) {
    const int byte_count = remaining_bits / 8;
    for(int i = 0; i < byte_count; i++) {
        const uint8_t value = src[i];
        dest[remaining_bytes++] = value;
        if(0xff == value) {
            dest[remaining_bytes++] = 0;
        }
    }
    remaining_bits &= 7;
    src[0] = src[byte_count];
}



__device__ void
gpujpeg_huffman_gpu_encoder_output(uint8_t * const src, int & remaining_bytes, uint8_t * & dest) {
    for(int i = 0; i < remaining_bytes; i++) {
        *(dest++) = src[i];
    }
    remaining_bytes = 0;
}


__device__ void
gpujpeg_huffman_gpu_encoder_emit_left_bits(uint8_t * &data_compressed, int * s_temp, int * s_out, int & remaining_bits, int & remaining_bytes, int tid) {
    if(tid == 0) {
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, (uint8_t*)s_temp, 7, 0x7f);
        gpujpeg_huffman_gpu_encoder_byte_stuff((uint8_t*)s_temp, remaining_bits, (uint8_t*)s_out, remaining_bytes);
        gpujpeg_huffman_gpu_encoder_output((uint8_t*)s_out, remaining_bytes, data_compressed);
    }
}


/**
 * Encode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ int
gpujpeg_huffman_gpu_encoder_encode_block(int16_t * block, uint8_t * &data_compressed, int * s_temp, int * s_out,
                int & remaining_bits, int & remaining_bytes, int *last_dc, int tid,
                struct gpujpeg_table_huffman_encoder* d_table_dc, struct gpujpeg_table_huffman_encoder* d_table_ac)
{
    // each thread loads a pair of values (pair after zigzag reordering)
    const int load_idx = tid * 2;
    int in_even = block[gpujpeg_huffman_gpu_encoder_order_natural[load_idx]];
    const int in_odd = block[gpujpeg_huffman_gpu_encoder_order_natural[load_idx + 1]];
    
    // compute number of zeros preceding the thread's even value
    const unsigned int even_nonzero_bitmap = 1 | __ballot(in_even); // DC coefficient is always treated as nonzero
    const unsigned int odd_nonzero_bitmap = __ballot(in_odd);
    const unsigned int nonzero_mask = (1 << tid) - 1;
    const int even_nonzero_count = __clz(even_nonzero_bitmap & nonzero_mask);
    const int odd_nonzero_count = __clz(odd_nonzero_bitmap & nonzero_mask);
    const int zeros_before_even = (min(odd_nonzero_count, even_nonzero_count) + tid - 32) * 2
                                + (odd_nonzero_count > even_nonzero_count ? 1 : 0);
    
    
    // true if any nonzero pixel follows thread's even pixel
    const unsigned int follow_mask = ~nonzero_mask;
    const bool nonzero_follows = follow_mask & ((even_nonzero_bitmap >> 1) | odd_nonzero_bitmap);
    
    // count of consecutive zeros before odd value (either one more than 
    // even if even is zero or none if even value itself is nonzero)
    const int zeros_before_odd = in_even || !tid ? 0 : zeros_before_even + 1;
    
    // (TODO: remove later) save leftover bits from previous iteration
    uint8_t leftover_value = *(uint8_t*)s_temp;
    
    // clear the buffer in parallel
    ((uint64_t*)s_temp)[tid] = 0;
    
    // put leftover value back
    if(0 == tid) {
        *(uint8_t*)s_temp = leftover_value;
    }
    
    
    
    
    // pointer to LUT for encoding thread's even value 
    // (only thread #0 uses DC table, others use AC table)
    const struct gpujpeg_table_huffman_encoder * d_table_even = d_table_ac;
    
    // first thread handles special DC coefficient
    if(0 == tid) {
        // first thread uses DC table for its even value
        d_table_even = d_table_dc;
        
        // update last DC coefficient
        const int original_in_even = in_even;
        in_even -= *last_dc;
        *last_dc = original_in_even;
    }
    
    // each thread encodes its two pixels
    int even_code_size = 0, even_code_value = 0, odd_code_size = 0, odd_code_value = 0;
    if(nonzero_follows || !tid) {
        gpujpeg_huffman_gpu_encode_value(even_code_size, even_code_value, zeros_before_even & 0xf, in_even, d_table_even);
        gpujpeg_huffman_gpu_encode_value(odd_code_size, odd_code_value, zeros_before_odd & 0xf, in_odd, d_table_ac);
    }
    if(tid == 31 && !in_odd) {
        odd_code_size = d_table_ac->size[256];
        odd_code_value = d_table_ac->code[256];
    }
    
    // replace values in shared memory with tuples (value, preceding zero count)
    s_temp[load_idx] = even_code_size + 32 * even_code_value;
    s_temp[load_idx + 1] = odd_code_size + 32 * odd_code_value;
    __threadfence_block();
    
    
    int result = 0;
    if(0 == tid) {
        for ( int k = 0; k < 64; k++ ) 
        {
            const int size = s_temp[k] & 31;
            const int value = s_temp[k] >> 5;
            if(k == 0) {
                *s_temp = leftover_value;   // put leftover bits back
            }
            gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, (uint8_t*)s_temp, size, value);
        }
        
        
        
        
        
        
        __threadfence_block();
        
        // apply byte stuffing to encoded bytes
        gpujpeg_huffman_gpu_encoder_byte_stuff((uint8_t*)s_temp, remaining_bits, (uint8_t*)s_out, remaining_bytes);
        
        __threadfence_block();
        
        gpujpeg_huffman_gpu_encoder_output((uint8_t*)s_out, remaining_bytes, data_compressed);
    }
    return __ballot(result);
}

/**
 * Huffman encoder kernel
 * 
 * @return void
 */
__global__ void
gpujpeg_huffman_encoder_encode_kernel(
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int comp_count,
    int segment_count, 
    uint8_t* d_data_compressed
#ifndef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
    ,struct gpujpeg_table_huffman_encoder* d_table_y_dc
    ,struct gpujpeg_table_huffman_encoder* d_table_y_ac
    ,struct gpujpeg_table_huffman_encoder* d_table_cbcr_dc
    ,struct gpujpeg_table_huffman_encoder* d_table_cbcr_ac
#endif
)
{    
#ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
    // Get huffman tables from constant memory
    struct gpujpeg_table_huffman_encoder* d_table_y_dc = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC];
    struct gpujpeg_table_huffman_encoder* d_table_y_ac = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC];
    struct gpujpeg_table_huffman_encoder* d_table_cbcr_dc = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC];
    struct gpujpeg_table_huffman_encoder* d_table_cbcr_ac = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC];
#endif
    
    int warpidx  = threadIdx.x >> 5;
    int tid    = threadIdx.x & 31;
    int remaining_bytes = 0;
    int remaining_bits = 0;

    __shared__ int s_temp_all[64 * WARPS_NUM];
    __shared__ int s_out_all[192 * WARPS_NUM];

    int * s_temp = s_temp_all + warpidx * 64;
    int * s_out  =  s_out_all + warpidx * 192;
    
    // Select Segment
    int segment_index = blockIdx.x * WARPS_NUM + warpidx;
    if ( segment_index >= segment_count )
        return;
    
    struct gpujpeg_segment* segment = &d_segment[segment_index];
    
    // Initialize huffman coder
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
        dc[comp] = 0;
    
    // Prepare data pointers
    uint8_t * data_compressed = &d_data_compressed[segment->data_compressed_index]; //TODO zmeni datovy typ
    uint8_t * data_compressed_start = data_compressed;
    
    // Non-interleaving mode
    if ( comp_count == 1 ) {

        // Get component for current scan
        struct gpujpeg_component* component = &d_component[segment->scan_index];

        // Get component data for MCU (first block)
        int16_t* block = &component->d_data_quantized[(segment_index * component->segment_mcu_count) * component->mcu_size];

        // Get coder parameters
        int & last_dc = dc[segment->scan_index];

        // Get huffman tables
        struct gpujpeg_table_huffman_encoder* d_table_dc = NULL;
        struct gpujpeg_table_huffman_encoder* d_table_ac = NULL;
        if ( component->type == GPUJPEG_COMPONENT_LUMINANCE ) {
            d_table_dc = d_table_y_dc;
            d_table_ac = d_table_y_ac;
        } else {
            d_table_dc = d_table_cbcr_dc;
            d_table_ac = d_table_cbcr_ac;
        }
            
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            // Encode 8x8 block
            if (gpujpeg_huffman_gpu_encoder_encode_block(block, data_compressed, s_temp, s_out, remaining_bits, remaining_bytes, &last_dc, tid, d_table_dc, d_table_ac) != 0)
                break;
            block += component->mcu_size;
        }
    }
    // Interleaving mode
    else {
        int segment_index = segment->scan_segment_index; //TODO asi nepotrebne
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            //assert(segment->scan_index == 0);
            for ( int comp = 0; comp < comp_count; comp++ ) {
                struct gpujpeg_component* component = &d_component[comp];

                // Prepare mcu indexes
                int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
                int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
                // Compute base data index
                int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                
                // For all vertical 8x8 blocks
                for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                    // Compute base row data index
                    int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                    // For all horizontal 8x8 blocks
                    for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                        // Compute 8x8 block data index
                        int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                        
                        // Get component data for MCU
                        int16_t* block = &component->d_data_quantized[data_index];
                        
                        // Get coder parameters
                        int & last_dc = dc[comp];
            
                        // Get huffman tables
                        struct gpujpeg_table_huffman_encoder* d_table_dc = NULL;
                        struct gpujpeg_table_huffman_encoder* d_table_ac = NULL;
                        if ( component->type == GPUJPEG_COMPONENT_LUMINANCE ) {
                            d_table_dc = d_table_y_dc;
                            d_table_ac = d_table_y_ac;
                        } else {
                            d_table_dc = d_table_cbcr_dc;
                            d_table_ac = d_table_cbcr_ac;
                        }
                        
                        // Encode 8x8 block
                        gpujpeg_huffman_gpu_encoder_encode_block(block, data_compressed, s_temp, s_out, remaining_bits, remaining_bytes, &last_dc, tid, d_table_dc, d_table_ac);
                    }
                }
            }
        }
    }

    // Emit left bits
    gpujpeg_huffman_gpu_encoder_emit_left_bits(data_compressed, s_temp, s_out, remaining_bits, remaining_bytes, tid);

    // Output restart marker
    if (tid == 0 ) {
        int restart_marker = GPUJPEG_MARKER_RST0 + (segment->scan_segment_index % 8);
        gpujpeg_huffman_gpu_encoder_marker(data_compressed, restart_marker);
                
        // Set compressed size
        segment->data_compressed_size = data_compressed - data_compressed_start;
    }
    __syncthreads();
}

/** Documented at declaration */
int
gpujpeg_huffman_gpu_encoder_init()
{
    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        (const char*)gpujpeg_huffman_gpu_encoder_order_natural,
        gpujpeg_order_natural, 
        GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );
    gpujpeg_cuda_check_error("Huffman encoder init");
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_huffman_gpu_encoder_encode(struct gpujpeg_encoder* encoder)
{    
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    
    assert(coder->param.restart_interval > 0);
    
    int comp_count = 1;
    if ( coder->param.interleaved == 1 )
        comp_count = coder->param_image.comp_count;
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);

    // Configure more shared memory
    cudaFuncSetCacheConfig(gpujpeg_huffman_encoder_encode_kernel, cudaFuncCachePreferShared);
            
    // Run kernel
    dim3 thread(32 * WARPS_NUM);
    dim3 grid(gpujpeg_div_and_round_up(coder->segment_count, (thread.x / 32)));
    gpujpeg_huffman_encoder_encode_kernel<<<grid, thread>>>(
        coder->d_component, 
        coder->d_segment, 
        comp_count,
        coder->segment_count, 
        coder->d_data_compressed
    #ifndef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC]
    #endif
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Huffman encoding failed");
    
    return 0;
}
