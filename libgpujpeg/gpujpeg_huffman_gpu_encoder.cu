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

/** Value decomposition in constant memory (input range from -4096 to 4095  ... both inclusive) */
__device__ unsigned int gpujpeg_huffman_value_decomposition[8 * 1024];

/** Size of occupied part of output buffer */
__device__ unsigned int gpujpeg_huffman_output_byte_count;

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


/** Initializes coefficient decomposition table in global memory */
static __global__ void
gpujpeg_huffman_gpu_encoder_value_decomposition_init_kernel() {
    // fetch some value
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int value = tid - 4096;
    
    // decompose it
    unsigned int value_code = value;
    int absolute = value;
    if ( value < 0 ) {
        // valu eis now absolute value of input
        absolute = -absolute;
        // For a negative input, want temp2 = bitwise complement of abs(input)
        // This code assumes we are on a two's complement machine
        value_code--;
    }

    // Find the number of bits needed for the magnitude of the coefficient
    unsigned int value_nbits = 0;
    while ( absolute ) {
        value_nbits++;
        absolute >>= 1;
    }
    
    // save result packed into unsigned int (value bits are left aligned in MSBs and size is right aligned in LSBs)
    gpujpeg_huffman_value_decomposition[tid] = value_nbits | (value_code << (32 - value_nbits));
//     printf("%+04d: %08x\n", value, gpujpeg_huffman_value_decomposition[tid]);
    
    // first thread also initializes size of final output, not to have to do it in separate memcpy/kernel
    if(0 == tid) {
        gpujpeg_huffman_output_byte_count = 0;
    }
}


/**
 * Adds up to 32 bits at once.
 * Codeword value must be aligned to left (most significant bits).
 */
__device__ inline void 
gpujpeg_huffman_gpu_encoder_emit_bits(unsigned int & remaining_bits, int & byte_count, int & bit_count, uint8_t * const out_ptr, const unsigned int packed_code_word)
{
    // decompose packed codeword into the msb-aligned value and bit-length of the value
    const unsigned int code_word = packed_code_word & ~31;
    const unsigned int code_bit_size = packed_code_word & 31;
    
    // concatenate with remaining bits
    remaining_bits |= code_word >> bit_count;
    bit_count += code_bit_size;
    if (bit_count >= 8) {
        do {
            const unsigned int out_byte = remaining_bits >> 24;
            out_ptr[byte_count++] = out_byte;
            if(0xff == out_byte) {
                out_ptr[byte_count++] = 0;
            }
            
            remaining_bits <<= 8;
            bit_count -= 8;
        } while (bit_count >= 8);
        
        remaining_bits = code_word << (code_bit_size - bit_count);
        remaining_bits &= 0xfffffffe << (31 - bit_count);
    }
}


__device__ static unsigned int
gpujpeg_huffman_gpu_encode_value(const int preceding_zero_count_idx, const int value,
                                 const struct gpujpeg_table_huffman_encoder * const d_table)
{
    // value bits are in MSBs (left aligned) and bit size of the value is in LSBs (right aligned)
    const unsigned int packed_value = gpujpeg_huffman_value_decomposition[4096 + value];
    
    // decompose value info into upshifted value and value's bit size
    const int value_nbits = packed_value & 0xf;
    const unsigned int value_code = packed_value & ~0xf;
    
    // find prefix of the codeword and size of the prefix
    const unsigned int prefix_code = d_table->gcode[preceding_zero_count_idx | value_nbits];
    const unsigned int prefix_nbits = prefix_code & 31;
    
    // compose packed codeword with its size
    return (prefix_code + value_nbits) | (value_code >> prefix_nbits);
}


__device__ void
gpujpeg_huffman_gpu_encoder_flush_codewords(unsigned int * const s_out, unsigned int * &data_compressed, int & remaining_codewords, const int tid) {
    // this works for up to 4 * 32 remaining codewords
    if(remaining_codewords) {
        // pad remianing codewords with extra zero-sized codewords, not to have to use special case in serialization kernel, which saves 4 codewords at once
        s_out[remaining_codewords + tid] = 0;
        
        // save all remaining codewords at once (together with some zero sized padding codewords)
        ((uint4*)data_compressed)[tid] = ((uint4*)s_out)[tid];
        
        // update codeword counter
        data_compressed += remaining_codewords;
        remaining_codewords = 0;
    }
}


/**
 * Encode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ int
gpujpeg_huffman_gpu_encoder_encode_block(const int16_t * block, unsigned int * &data_compressed, unsigned int * const s_out,
                int & remaining_codewords, int *last_dc, int tid,
                struct gpujpeg_table_huffman_encoder* d_table_dc, struct gpujpeg_table_huffman_encoder* d_table_ac)
{
    // each thread loads a pair of values (pair after zigzag reordering)
    const int load_idx = tid * 2;
    int in_even = block[gpujpeg_huffman_gpu_encoder_order_natural[load_idx]];
    const int in_odd = block[gpujpeg_huffman_gpu_encoder_order_natural[load_idx + 1]];
    
    // compute preceding zero count for even coefficient (actually compute the count multiplied by 16)
    const unsigned int nonzero_mask = (1 << tid) - 1;
    const unsigned int nonzero_bitmap_0 = 1 | __ballot(in_even);  // DC is always treated as nonzero
    const unsigned int nonzero_bitmap_1 = __ballot(in_odd);
    const unsigned int nonzero_bitmap_pairs = nonzero_bitmap_0 | nonzero_bitmap_1;
    
    const int zero_pair_count = __clz(nonzero_bitmap_pairs & nonzero_mask);
    int zeros_before_even = 32 * (zero_pair_count + tid - 32);
    if((0x80000000 >> zero_pair_count) > (nonzero_bitmap_1 & nonzero_mask)) {
        zeros_before_even += 16;
    }
    
    // true if any nonzero pixel follows thread's odd pixel
    const bool nonzero_follows = nonzero_bitmap_pairs & ~nonzero_mask;
    
    // count of consecutive zeros before odd value (either one more than 
    // even if even is zero or none if even value itself is nonzero)
    // (the count is actually multiplied by 16)
    int zeros_before_odd = in_even || !tid ? 0 : zeros_before_even + 16;
    
    // clear zero counts if no nonzero pixel follows (so that no 16-zero symbols will be emited)
    // otherwise only trim extra bits from the counts of following zeros
    const int zero_count_mask = nonzero_follows ? 0xF0 : 0;
    zeros_before_even &= zero_count_mask;
    zeros_before_odd &= zero_count_mask;
    
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
    
    // last thread handles special block-termination symbol
    if(0 == ((tid ^ 31) | in_odd)) {
        // this causes selection of huffman symbol at index 256 (which contains the termination symbol)
        zeros_before_odd = 256;
    }
    
    // each thread gets codeword for its two pixels
    unsigned int even_code = gpujpeg_huffman_gpu_encode_value(zeros_before_even, in_even, d_table_even);
    unsigned int odd_code = gpujpeg_huffman_gpu_encode_value(zeros_before_odd, in_odd, d_table_ac);
    
    // concatenate both codewords into one if they are short enough
    const unsigned int even_code_size = even_code & 31;
    const unsigned int odd_code_size = odd_code & 31;
    const unsigned int total_size = even_code_size + odd_code_size;
    if(total_size <= 27) {
        even_code = total_size | ((odd_code & ~31) >> even_code_size) | (even_code & ~31);
        odd_code = 0;
    }
    
    // each thread get number of preceding nonzero codewords and total number of nonzero codewords in this block
    const unsigned int even_codeword_presence = __ballot(even_code);
    const unsigned int odd_codeword_presence = __ballot(odd_code);
    const int codeword_offset = __popc(nonzero_mask & even_codeword_presence)
                              + __popc(nonzero_mask & odd_codeword_presence);
    
    // each thread saves its values into temporary shared buffer
    if(even_code) {
        s_out[remaining_codewords + codeword_offset] = even_code;
        if(odd_code) {
            s_out[remaining_codewords + codeword_offset + 1] = odd_code;
        }
    }
    
    // advance count of codewords in shared memory buffer
    remaining_codewords += __popc(odd_codeword_presence) + __popc(even_codeword_presence);
    
    // flush some codewords to global memory if there are too many of them in shared buffer
    const int flush_count = 32 * 4; // = half of the buffer
    if(remaining_codewords > flush_count) {
        // move first half of the buffer into output buffer in global memory and update output pointer
        ((uint4*)data_compressed)[tid] = ((uint4*)s_out)[tid];
        data_compressed += flush_count;
        
        // shift remaining codewords to begin of the buffer and update their count
        ((uint4*)s_out)[tid] = ((uint4*)s_out)[flush_count / 4 + tid];  // 4 for 4 uints in uint4
        remaining_codewords -= flush_count;
    }
        
    // nothing to fail here
    return 0;
}




/**
 * Huffman encoder kernel
 * 
 * @return void
 */
__launch_bounds__(WARPS_NUM * 32, 1024 / (WARPS_NUM * 32))
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
    
    int warpidx = threadIdx.x >> 5;
    int tid = threadIdx.x & 31;

    __shared__ uint4 s_out_all[64 * WARPS_NUM];
    unsigned int * s_out = (unsigned int*)(s_out_all + warpidx * 64);
    
    // Number of remaining codewords in shared buffer
    int remaining_codewords = 0;
    
    // Select Segment
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int segment_index = block_idx * WARPS_NUM + warpidx;
    if ( segment_index >= segment_count )
        return;
    
    struct gpujpeg_segment* segment = &d_segment[segment_index];
    const int segment_mcu_count = segment->mcu_count;
    
    // Initialize huffman coder
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
        dc[comp] = 0;
    
    // Prepare data pointers
    unsigned int * data_compressed = (unsigned int*)(d_data_compressed + segment->data_temp_index);
    unsigned int * data_compressed_start = data_compressed;
    
    // Non-interleaving mode
    if ( comp_count == 1 ) {
        int segment_index = segment->scan_segment_index;

        // Get component for current scan
        struct gpujpeg_component* component = &d_component[segment->scan_index];

        // Get component data for MCU (first block)
        const int16_t* block = &component->d_data_quantized[(segment_index * component->segment_mcu_count) * component->mcu_size];

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
        
        // mcu size of the component
        const int comp_mcu_size = component->mcu_size;
            
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment_mcu_count; mcu_index++ ) {
            // Encode 8x8 block
            if (gpujpeg_huffman_gpu_encoder_encode_block(block, data_compressed, s_out, remaining_codewords, &last_dc, tid, d_table_dc, d_table_ac) != 0)
                break;
            block += comp_mcu_size;
        }
    }
    // Interleaving mode
    else {
        int segment_index = segment->scan_segment_index; //TODO asi nepotrebne
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment_mcu_count; mcu_index++ ) {
            //assert(segment->scan_index == 0);
            for ( int comp = 0; comp < comp_count; comp++ ) {
                struct gpujpeg_component* component = &d_component[comp];

                // Prepare mcu indexes
                int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
                int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
                
                // Compute base data index
                int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                
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
                
                // Get sampling faactors
                const int comp_vertical_sampling = component->sampling_factor.vertical;
                const int comp_horizontal_sampling = component->sampling_factor.horizontal;
                
                // Pointer to quantized data of the component
                const int16_t* const d_component_data_quantized = component->d_data_quantized;
                
                // Size and count of MCUs in component
                const int comp_mcu_stride = component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE;
                
                // For all vertical 8x8 blocks
                for ( int y = 0; y < comp_vertical_sampling; y++ ) {
                    // Compute base row data index
                    int data_index_row = data_index_base + y * comp_mcu_stride;
                    // For all horizontal 8x8 blocks
                    for ( int x = 0; x < comp_horizontal_sampling; x++ ) {
                        // Compute 8x8 block data index
                        int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                        
                        // Get component data for MCU
                        const int16_t* block = d_component_data_quantized + data_index;
                        
                        // Get coder parameters
                        int & last_dc = dc[comp];
            
                        // Encode 8x8 block
                        gpujpeg_huffman_gpu_encoder_encode_block(block, data_compressed, s_out, remaining_codewords, &last_dc, tid, d_table_dc, d_table_ac);
                    }
                }
            }
        }
    }

    // flush remaining codewords
    gpujpeg_huffman_gpu_encoder_flush_codewords(s_out, data_compressed, remaining_codewords, tid);
    
    // Set number of codewords.
    if (tid == 0 ) {
        segment->data_compressed_size = data_compressed - data_compressed_start;
    }
    __syncthreads();
}



#define SERIALIZATION_THREADS_PER_TBLOCK 192


/**
 * Codeword serialization kernel.
 * 
 * @return void
 */
__launch_bounds__(SERIALIZATION_THREADS_PER_TBLOCK, 1536 / SERIALIZATION_THREADS_PER_TBLOCK)
__global__ static void
gpujpeg_huffman_encoder_serialization_kernel(
    struct gpujpeg_segment* d_segment,
    int segment_count, 
    uint8_t* d_data_compressed
) {    
    // Temp buffer for all threads of the threadblock
    __shared__ uint4 s_temp_all[2 * SERIALIZATION_THREADS_PER_TBLOCK];

    // Thread's 32 bytes in shared memory for output composition
    uint4 * const s_temp = s_temp_all + threadIdx.x * 2;
    
    // Select Segment
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int segment_index = block_idx * SERIALIZATION_THREADS_PER_TBLOCK + threadIdx.x;
    if ( segment_index >= segment_count )
        return;
    
    // Thread's segment
    struct gpujpeg_segment* const segment = &d_segment[segment_index];
    
    // Input and output pointers
    uint4 * const d_dest_stream_start = (uint4*)(d_data_compressed + segment->data_temp_index);
    uint4 * d_dest_stream = d_dest_stream_start;
    const uint4 * d_src_codewords = d_dest_stream_start;
    
    // number of bytes in the temp buffer, remaining bits and their count
    int byte_count = 0, bit_count = 0;
    unsigned int remaining_bits = 0;
    
    // "data_compressed_size" is now initialized to number of codewords to be serialized
    for(int cword_tuple_count = (segment->data_compressed_size + 3) >> 2; cword_tuple_count--; ) // reading 4 codewords at once
    {
        // read 4 codewords and advance input pointer to next ones
        const uint4 cwords = *(d_src_codewords++);
        
        // encode first pair of codewords
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, cwords.x);
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, cwords.y);
        
        // possibly flush output if have at least 16 bytes
        if(byte_count >= 16) {
            // write 16 bytes into destination buffer
            *(d_dest_stream++) = s_temp[0];
            
            // move remaining bytes to first half of the buffer
            s_temp[0] = s_temp[1];
            
            // update number of remaining bits
            byte_count -= 16;
        }
        
        // encode other two codewords
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, cwords.z);
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, cwords.w);
        
        // possibly flush output if have at least 16 bytes
        if(byte_count >= 16) {
            // write 16 bytes into destination buffer
            *(d_dest_stream++) = s_temp[0];
            
            // move remaining bytes to first half of the buffer
            s_temp[0] = s_temp[1];
            
            // update number of remaining bits
            byte_count -= 16;
        }
    }
    
    // Emit left bits
    gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, 0xfe000007);

    // Terminate codestream with restart marker
    ((uint8_t*)s_temp)[byte_count + 0] = 0xFF;
    ((uint8_t*)s_temp)[byte_count + 1] = GPUJPEG_MARKER_RST0 + (segment->scan_segment_index % 8);
    
    // flush remaining bytes
    d_dest_stream[0] = s_temp[0];
    d_dest_stream[1] = s_temp[1];
    
    // Set compressed size
    segment->data_compressed_size = (d_dest_stream - d_dest_stream_start) * 16 + byte_count + 2;
}




/**
 * Huffman coder output compaction kernel.
 * 
 * @return void
 */
__global__ static void
gpujpeg_huffman_encoder_compaction_kernel (
    struct gpujpeg_segment* const d_segment,
    const int segment_count, 
    const uint8_t* const d_src,
    uint8_t* const d_dest
) {    
    // get some segment (size of threadblocks is 32 x N, so threadIdx.y is warp index)
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int segment_idx = threadIdx.y + block_idx * blockDim.y;
    if(segment_idx >= segment_count) {
        return;
    }
    
    // temp variables for all warps
    __shared__ volatile unsigned int s_out_offsets[WARPS_NUM];
    
    // get info about the segment
    const unsigned int segment_byte_count = (d_segment[segment_idx].data_compressed_size + 15) & ~15;  // number of bytes rounded up to multiple of 16
    const unsigned int segment_in_offset = d_segment[segment_idx].data_temp_index;  // this should be aligned at least to 16byte boundary
    
    // first thread of each warp reserves space in output buffer
    if(0 == threadIdx.x) {
         const unsigned int segment_out_offset = atomicAdd(&gpujpeg_huffman_output_byte_count, segment_byte_count);
         s_out_offsets[threadIdx.y] = segment_out_offset;
         d_segment[segment_idx].data_compressed_index = segment_out_offset;
    }
    
    // all threads read output buffer offset for their segment and prepare input and output pointers and number of copy iterations
    const uint4 * d_in = threadIdx.x + (uint4*)(d_src + segment_in_offset);
    uint4 * d_out = threadIdx.x + (uint4*)(d_dest + s_out_offsets[threadIdx.y]);
    unsigned int copy_iterations = segment_byte_count / 512; // 512 is number of bytes copied in each iteration (32 threads * 16 bytes per thread)
    
    // copy the data!
    while(copy_iterations--) {
        *d_out = *d_in;
        d_out += 32;
        d_in += 32;
    }
    
    // copy remaining bytes (less than 512 bytes)
    if((threadIdx.x * 16) < (segment_byte_count & 511)) {
        *d_out = *d_in;
    }
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


dim3
gpujpeg_huffman_gpu_encoder_grid_size(int tblock_count)
{
    dim3 size(tblock_count);
    while(size.x > 0xffff) {
        size.x = (size.x + 1) >> 1;
        size.y <<= 1;
    }
    return size;
}




/** Documented at declaration */
int
gpujpeg_huffman_gpu_encoder_encode(struct gpujpeg_encoder* encoder, unsigned int * output_byte_count)
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
    cudaFuncSetCacheConfig(gpujpeg_huffman_encoder_serialization_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_gpu_encoder_value_decomposition_init_kernel, cudaFuncCachePreferShared);
    
    // Initialize decomposition lookup table
    gpujpeg_huffman_gpu_encoder_value_decomposition_init_kernel<<<32, 256>>>();  // 8192 threads total
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Decomposition LUT initialization failed");
    
    // Run encoder kernel
    dim3 thread(32 * WARPS_NUM);
    dim3 grid = gpujpeg_huffman_gpu_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, (thread.x / 32)));
    gpujpeg_huffman_encoder_encode_kernel<<<grid, thread>>>(
        coder->d_component, 
        coder->d_segment, 
        comp_count,
        coder->segment_count, 
        coder->d_temp_huffman
    #ifndef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC]
    #endif
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Huffman encoding failed");
    
    
    // Run codeword serialization kernel
    const int num_serialization_tblocks = gpujpeg_div_and_round_up(coder->segment_count, SERIALIZATION_THREADS_PER_TBLOCK);
    const dim3 serialization_grid = gpujpeg_huffman_gpu_encoder_grid_size(num_serialization_tblocks);
    gpujpeg_huffman_encoder_serialization_kernel<<<num_serialization_tblocks, SERIALIZATION_THREADS_PER_TBLOCK>>>(
        coder->d_segment, 
        coder->segment_count, 
        coder->d_temp_huffman
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Codeword serialization failed");
    
    // Run output compaction kernel (one warp per segment)
    const dim3 compaction_thread(32, WARPS_NUM);
    const dim3 compaction_grid = gpujpeg_huffman_gpu_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, WARPS_NUM));
    gpujpeg_huffman_encoder_compaction_kernel<<<compaction_grid, compaction_thread>>>(
        coder->d_segment,
        coder->segment_count,
        coder->d_temp_huffman,
        coder->d_data_compressed
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Huffman output compaction failed");
    
    // Read and return number of occupied bytes
    cudaMemcpyFromSymbol(output_byte_count, gpujpeg_huffman_output_byte_count, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    gpujpeg_cuda_check_error("Huffman output size getting failed");
    
    return 0;
}
