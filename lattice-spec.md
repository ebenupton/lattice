# Lattice Video Codec Specification

**Draft 0.3.3 — February 2026**

---

## 1. Introduction

### 1.1 Scope

This document specifies Lattice, a lossy video codec for consumer video applications. The specification defines a bitstream syntax and a normative decoding process: a conformant decoder presented with a valid Lattice bitstream shall produce a unique sequence of decoded frames, bit-identical across all conformant implementations.

Lattice targets bitrates from 1–100 Mbps and resolutions from 720p to 4K at frame rates up to 60 fps. It is designed for playback on mainstream consumer devices manufactured from the mid-2020s onward.

### 1.2 Goals

Lattice has three primary goals, in order of priority:

1. **Implementation simplicity.** A competent engineer should be able to read this specification and write a conformant decoder in a matter of weeks, not months or years. The reference decoder is approximately 1,200 lines of C. The codec achieves this by using a small number of powerful, general-purpose tools rather than a large number of specialised ones.

2. **Compression efficiency.** Lattice targets compression performance competitive with contemporary codecs (VVC, AV2) across its operating range. It accepts that individual tools may be suboptimal in isolation, provided the system as a whole performs well. In particular, it relies on an adaptive convolutional neural network loop filter to compensate for deliberate simplifications elsewhere in the pipeline.

3. **Freedom from encumbrance.** Lattice avoids the use of techniques known to be covered by active patents. Where a patented technique offers marginal gains, Lattice substitutes a simpler alternative and recovers the difference through its adaptive loop filter.

### 1.3 Design Philosophy

#### 1.3.1 Target Implementations

The design of Lattice is informed by three concrete target implementations, which represent the range of hardware on which consumer video decoding takes place:

**Software on a mobile application processor.** Two cores of an Arm Cortex-A76 class CPU (or equivalent), with 128-bit SIMD (NEON/SVE). This is the minimum viable implementation. The entire decoder — entropy decoding, prediction, transform, reconstruction, and loop filtering — runs on the CPU. Key constraints: limited multiply throughput (~20 GMAC/s per core for int16×int16), 64 KB L1 data cache per core, and sequential entropy decoding as the throughput bottleneck.

**Software on a mobile application processor with GPU assistance.** The same CPU handles bitstream parsing and entropy decoding. A mobile-class GPU (approximately 50 GFLOPS theoretical, 2 Gtexel/s texture lookup rate — comparable to an Arm Mali-G76 or similar) handles motion compensation, inverse transform, and loop filtering as compute shader dispatches. Key constraints: GPU integer arithmetic must be bit-exact (ruling out texture filtering hardware for normative operations), and all pixel-plane workloads must fit within the GPU's compute budget.

**Dedicated hardware accelerator.** A small fixed-function decoder implemented at approximately 0.4 mm² in a 16 nm process. Contains a dual-stream entropy decode engine, an inverse transform unit, a motion compensation unit, and a shared multiply-accumulate array for neural network operations. Key constraints: all buffer sizes must be statically determinable, and the MAC array is time-shared between codec stages.

These three targets impose specific, sometimes competing constraints on the codec design. Where they conflict, the specification documents the trade-off. The general principle is: **the bitstream syntax is designed for the hardware decoder, the arithmetic is designed for the GPU, and the complexity budget is designed for the software decoder.**

#### 1.3.2 Two-Plane Architecture

Lattice cleanly separates *structural decoding* (bitstream parsing, entropy decoding, block map construction, motion vector recovery) from *pixel-domain reconstruction* (prediction, transform, filtering). These are referred to as the **decision plane** and the **pixel plane** respectively.

The decision plane is inherently sequential: each symbol's context depends on previously decoded symbols. It is designed to run on a CPU or a small fixed-function state machine, and its throughput determines the codec's serial bottleneck.

The pixel plane is inherently parallel: once a block's prediction mode, motion vectors, and transform coefficients are known, its reconstruction is independent of other blocks within a tile. It is designed to run on a GPU's compute shaders or on the hardware accelerator's datapath units.

This separation is not merely an implementation convenience — it is a structural property of the bitstream. The decision plane can run ahead of the pixel plane, filling a work queue of block descriptors that the pixel plane consumes asynchronously.

#### 1.3.3 Simplicity Through Generality

Traditional video codecs achieve high compression through a large collection of specialised tools: many intra prediction modes, multiple interpolation filter sets, cascaded loop filters, secondary transforms, and so on. Each tool is individually optimised for a narrow class of content, and the encoder selects among them per block.

Lattice takes a different approach. It uses a small number of general-purpose tools:

- **One intra prediction mode** (parametric gradient), derived deterministically from reconstructed neighbours with no signalling cost.
- **One interpolation filter** (integer bilinear), the simplest possible sub-pixel filter.
- **One loop filter** (a 4-layer convolutional neural network), whose weights are signalled per frame and which subsumes the roles of deblocking, sample-adaptive offset, and adaptive loop filtering.

The CNN loop filter is the key enabler of this philosophy. Because its weights are adapted per frame, it can learn to compensate for the specific artefacts produced by the codec's deliberately simplified prediction and interpolation tools. The encoder trains the filter weights (using any method it chooses — this is outside the scope of this specification) to minimise distortion on the current frame. The decoder simply applies them.

This design trades per-tool optimality for system-level simplicity. The total decoder complexity is far lower than codecs of comparable efficiency, because most of the "intelligence" resides in the encoder's choice of filter weights rather than in the decoder's tool repertoire.

---

## 2. Conventions and Definitions

### 2.1 Terminology

**Shall** indicates a normative requirement on a conformant decoder.

**Tile**: A 128×128 sample region of the luma plane (and corresponding 64×64 region of each chroma plane in 4:2:0). Tiles are independently decodable after parsing the frame header.

**Cell**: An 8×8 sample region within a tile. Cells are the atomic unit of the block partitioning grid.

**Block**: A rectangular region composed of one or more merged cells. Blocks are the unit of prediction, transform, and coefficient coding.

**Band**: A contiguous group of transform coefficients at similar frequencies, used as the unit of coefficient coding.

**Decision plane**: The portion of the decoding process concerned with bitstream parsing, entropy decoding, and syntax element recovery.

**Pixel plane**: The portion of the decoding process concerned with prediction, reconstruction, and filtering of sample values.

### 2.2 Arithmetic Conventions

All arithmetic operations in this specification are integer operations with defined bit widths and rounding behaviour. There is no floating-point arithmetic in the normative decoding process.

- **intN**: A signed two's complement integer of N bits.
- **uintN**: An unsigned integer of N bits.
- **Multiplication**: Unless otherwise stated, multiplication of an intA by an intB or uintB produces a result with sufficient precision to hold the exact product (typically int32 or int64 as specified per operation).
- **Right shift**: Arithmetic right shift (sign-extending). Written as `>> n`.
- **Rounding right shift**: `round_shift(x, n)` computes `(x + (1 << (n-1))) >> n`.
- **Clamp**: `clamp(x, lo, hi)` returns `max(lo, min(hi, x))`.

### 2.3 Sample Bit Depth

This specification defines decoding for 8-bit and 10-bit sample depths. The internal processing bit depth (referred to as `BitDepth`) is either 8 or 10, signalled in the sequence header. All sample values are unsigned integers in the range [0, 2^BitDepth − 1].

### 2.4 Byte Order and Bitstream Reading

All multi-byte fixed-width fields in headers (sequence header, frame header, tile header) are stored in **big-endian** byte order (most significant byte first).

Bits within the bypass region are read most-significant-bit first from each byte. The bypass reader maintains a byte pointer and a bit index (7 = MSB, 0 = LSB). Reading one bit:

```
bit = (byte[byte_pos] >> bit_idx) & 1
bit_idx -= 1
if bit_idx < 0:
    bit_idx = 7
    byte_pos += 1
```

### 2.5 Colour Space

Lattice operates in the YCbCr colour space with BT.709 transfer characteristics. The luma plane (Y) and two chroma planes (Cb, Cr) are coded independently. Conversion to and from RGB is outside the scope of this specification.

---

## 3. Bitstream Structure

### 3.1 Overview

A Lattice bitstream consists of a sequence header followed by a series of coded frames. Each coded frame begins with a frame header, followed by a sequence of independently coded tiles.

```
Sequence:
  sequence_header
  coded_frame[0]
  coded_frame[1]
  ...

Coded frame:
  frame_header
  tile_data[0]
  tile_data[1]
  ...
```

### 3.2 Sequence Header

The sequence header signals parameters that are constant for the duration of the coded sequence. Fields are byte-aligned and read in order:

| Field | Type | Description |
|-------|------|-------------|
| `magic` | uint32 | Magic number: 0x4C415454 ("LATT") |
| `frame_width` | uint16 | Luma frame width in samples |
| `frame_height` | uint16 | Luma frame height in samples |
| `bit_depth` | uint8 | 8 or 10 |
| `max_ref_frames` | uint8 | Maximum number of reference frames (1–8) |

Total sequence header size: 10 bytes.

The number of tiles in the horizontal and vertical directions is derived:

```
tiles_wide = (frame_width + 127) / 128      // integer ceiling division
tiles_high = (frame_height + 127) / 128
```

Tiles at the right and bottom frame edges may be smaller than 128×128. These **partial tiles** are coded identically to full tiles, except that cells falling entirely outside the frame boundary are omitted from the block map and do not generate output samples.

The actual tile dimensions for the tile at column `tx`, row `ty` are:

```
tile_w = min(128, frame_width  - tx * 128)
tile_h = min(128, frame_height - ty * 128)
tile_cells_w = (tile_w + 7) / 8
tile_cells_h = (tile_h + 7) / 8
```

### 3.3 Frame Header

The frame header signals per-frame parameters. Fields are byte-aligned:

| Field | Type | Description |
|-------|------|-------------|
| `frame_type` | uint8 | 0 = intra frame (I), 1 = inter frame (P) |
| `base_qp` | uint8 | Base quantisation parameter (0–51) |
| `filter_mode` | uint8 | 0 = default filter weights, 1 = custom luma weights follow |
| `filter_rans_size` | uint16 | Size in bytes of rANS-coded filter weight data (present only if `filter_mode` = 1) |
| `filter_rans_data` | uint8[filter_rans_size] | rANS-coded filter weight deltas (present only if `filter_mode` = 1; see §8.5) |

On intra frames, no reference frames are used. On inter frames, all frames currently in the DPB are available as references (the decoder tracks DPB occupancy; see §11).

### 3.4 Tile Data

Each tile's coded data is laid out as four contiguous regions:

```
tile_data:
  tile_header         (5 bytes, byte-aligned)
  rans_stream_0       (read forward from byte 0 of payload)
  rans_stream_1       (read backward from byte bypass_offset-1 of payload)
  bypass_bits         (read forward from byte bypass_offset of payload)
```

The tile header contains:

| Field | Type | Description |
|-------|------|-------------|
| `tile_data_size` | uint24 | Total size of this tile's payload in bytes (after header) |
| `bypass_offset` | uint16 | Byte offset within payload where the bypass region begins |

The **payload** is the `tile_data_size` bytes immediately following the tile header. Within the payload:

- rANS stream 0 starts at payload byte 0 and reads forward.
- rANS stream 1 starts at payload byte `bypass_offset − 1` and reads backward.
- The bypass region starts at payload byte `bypass_offset` and reads forward.

The two rANS streams occupy the bytes [0, `bypass_offset`). Stream 0 consumes bytes from the low end; stream 1 consumes bytes from the high end. They grow toward each other and shall not overlap — a conformant bitstream guarantees that the total bytes consumed by both streams does not exceed `bypass_offset`.

### 3.5 rANS Stream Initialisation

Each rANS stream's 32-bit state is initialised by reading 4 bytes in big-endian order:

```
// Stream 0 (forward):
x0 = (payload[0] << 24) | (payload[1] << 16) | (payload[2] << 8) | payload[3]
stream0_pos = 4    // next byte to read

// Stream 1 (backward):
p = bypass_offset - 1
x1 = (payload[p] << 24) | (payload[p-1] << 16) | (payload[p-2] << 8) | payload[p-3]
stream1_pos = p - 4    // next byte to read (moving backward)
```

The initial state values shall be in the range [2^16, 2^32). A conformant encoder ensures this.

---

## 4. Block Partitioning

### 4.1 Design Rationale

Block partitioning in traditional codecs uses recursive tree structures: quadtrees (HEVC), multi-type trees (VVC), or superblock recursion (AV1). These achieve fine-grained adaptation at the cost of complex tree-walking logic in the decoder and variable, data-dependent memory access patterns in the pixel plane.

Lattice uses a flat grid-merge scheme. Each tile is divided into a fixed grid of 8×8 cells (up to 16×16 cells for a full tile). Blocks are formed by merging rectangular groups of cells, subject to power-of-two constraints. The result is a flat list of non-overlapping blocks that cover the tile — no tree traversal required. The decoder reads the block map, builds an array of block descriptors (position, size, shape), and dispatches them to the pixel plane.

This design is motivated by the hardware decoder target, where a flat block list maps directly to a simple dispatch FIFO, and by the GPU target, where a flat list of independent work items maps to a compute shader dispatch with no control flow divergence.

### 4.2 Block Shapes

Block dimensions are constrained to power-of-two multiples of the cell size, with a maximum aspect ratio of 2:1. The permitted block shapes are:

| Shape Index | Width (samples) | Height (samples) | Width (cells) | Height (cells) |
|-------------|----------------|-----------------|---------------|----------------|
| 0 | 8 | 8 | 1 | 1 |
| 1 | 16 | 8 | 2 | 1 |
| 2 | 8 | 16 | 1 | 2 |
| 3 | 16 | 16 | 2 | 2 |
| 4 | 32 | 16 | 4 | 2 |
| 5 | 16 | 32 | 2 | 4 |
| 6 | 32 | 32 | 4 | 4 |

Power-of-two dimensions ensure that every block aligns naturally to the cell grid, that transform sizes always match block dimensions without remainder, and that the block map is equivalent to a constrained quadtree with optional pair-merging — simplifying both encoder search and decoder validation.

The maximum block size of 32×32 eliminates the need for a 64-point transform, reducing the transform library to three sizes (8, 16, and 32 points), plus a 4-point transform for chroma blocks derived from 8×8 luma blocks.

### 4.3 Block Map Coding

The block map is coded cell-by-cell in raster order (left to right, top to bottom) within each tile. At each cell position, the decoder determines whether this cell begins a new block or is interior to an already-established block.

For each cell that begins a new block, the decoder reads a `shape_index` symbol (0–6) from the full 7-symbol context-coded CDF. The same 7-entry CDF is used at every cell position regardless of which shapes fit — no alphabet reduction or CDF modification is performed. CDF adaptation (§7.4) updates the full 7-entry CDF after each decoded symbol in the standard way. The CDF naturally suppresses shapes that the encoder never selects near tile boundaries.

The context for `shape_index` is derived from the blocks above and to the left:

```
size_category(block_area):
    if block_area <= 64:   return 0    // 8x8 only (small)
    if block_area <= 256:  return 1    // 16x8 through 16x16 (medium)
    return 2                            // 32x16 through 32x32 (large)

above_cat = size_category(block_above.area) if above exists else 0
left_cat  = size_category(block_left.area)  if left exists  else 0
context_index = above_cat * 3 + left_cat    // 0..8
```

Context index 0–8 selects among 9 context slots for the shape symbol CDF.

Cells that are interior to an already-established block (i.e., cells whose position falls within the bounds of a previously decoded block) are skipped — no symbol is coded for them.

The decoder maintains a `tile_cells_w × tile_cells_h` array of block indices. When a new block is decoded at cell position (cx, cy) with dimensions (w_cells, h_cells), all cells in the rectangle [cx, cx+w_cells) × [cy, cy+h_cells) are marked as belonging to that block. Subsequent cells in raster order that are already marked are skipped.

After decoding `shape_index`, the decoder shall verify that the shape fits within the remaining tile space at the current cell position. A conformant bitstream shall not specify blocks that extend beyond the tile boundary or that overlap with previously established blocks. The decoder shall reject non-conformant bitstreams.

---

## 5. Block Decoding Order

### 5.1 Per-Block Syntax

After the block map is decoded for a tile, the decoder processes each block in raster order (the order in which their top-left cells appear in the cell raster scan). For each block, the following syntax elements are decoded in order:

1. **Prediction mode** (§5.2)
2. **Reference index** (§5.4.1), if inter-predicted and DPB count > 1
3. **Motion vector delta** (§5.4.2), if inter-predicted (modes 1)
4. **QP delta** (§6.3), if not SKIP
5. **Coded block flag (CBF)** (§5.1.1), if not SKIP
6. **Coefficients** (§9), if CBF = 1

For SKIP mode blocks (prediction mode 2), steps 3–6 are omitted: the MV is the predictor (§5.4.3), QP delta is 0, CBF is implicitly 0, and no coefficients are coded.

#### 5.1.1 Coded Block Flag

The coded block flag (CBF) is a binary symbol indicating whether the block has any non-zero residual coefficients.

```
cbf_context = (above_block.cbf if above exists else 0)
            + (left_block.cbf  if left exists  else 0)
// cbf_context is 0, 1, or 2
```

CBF is decoded from a 2-symbol alphabet using context slot `CBF_BASE + cbf_context` (3 context slots total).

If CBF = 0, the residual for this block is all zeros: the reconstructed samples equal the prediction. If CBF = 1, luma and chroma coefficients are decoded as specified in §9.

### 5.2 Prediction Modes

Each block is assigned a prediction mode, coded as a single symbol:

| Mode Index | Name | Description |
|-----------|------|-------------|
| 0 | `INTRA` | Gradient plane prediction from reconstructed neighbours |
| 1 | `INTER` | Motion-compensated prediction from one reference frame |
| 2 | `SKIP` | Copy from reference using MV predictor, no residual |

On intra frames (`frame_type` = 0), all blocks are `INTRA`; the mode symbol is not coded.

On inter frames, the mode is decoded from a 3-entry alphabet with context:

```
mode_category(mode):
    if mode == 0: return 0    // intra
    if mode == 1: return 1    // inter
    return 2                   // skip

above_cat = mode_category(block_above.mode) if above exists else 1
left_cat  = mode_category(block_left.mode)  if left exists  else 1
context_index = above_cat * 3 + left_cat    // 0..8
```

#### 5.2.1 SKIP Mode

SKIP mode performs motion-compensated prediction using the MV predictor (§5.4.3) with no delta, from reference index 0 (the most recent frame in the DPB). No MV delta, QP delta, or coefficients are coded.

### 5.3 Intra Prediction: Parametric Gradient

The intra predictor generates a prediction for the current block by fitting a plane model to the reconstructed samples immediately above and to the left of the block. The model has three parameters: a DC offset, a horizontal gradient, and a vertical gradient.

The rationale for this choice is simplicity. The parameters are derived deterministically from available neighbours, requiring no signalling. A plane model is robust and well-conditioned when only top and left neighbours are available. Content that a plane model handles poorly — textures, sharp edges — is handled by the residual signal and the CNN loop filter, not by a larger intra mode repertoire.

#### 5.3.1 Parameter Derivation

Let the current block have width W and height H samples. Let T[x] for x in [0, W) denote the reconstructed samples immediately above the block (the bottom row of the block above), and L[y] for y in [0, H) denote the reconstructed samples immediately to the left (the rightmost column of the block to the left).

If both top and left neighbours are available:

```
dc = round_div(sum(T[0..W-1]) + sum(L[0..H-1]), W + H)
delta_h = T[W-1] - T[0]
delta_v = L[H-1] - L[0]
```

If only the top neighbour is available (block at left tile edge):

```
dc = round_div(sum(T[0..W-1]), W)
delta_h = T[W-1] - T[0]
delta_v = 0
```

If only the left neighbour is available (block at top tile edge):

```
dc = round_div(sum(L[0..H-1]), H)
delta_h = 0
delta_v = L[H-1] - L[0]
```

If neither neighbour is available (top-left block of the tile):

```
dc = 1 << (BitDepth - 1)
delta_h = 0
delta_v = 0
```

The `round_div` function is defined in Appendix A.

#### 5.3.2 Prediction Generation

The predicted sample at position (x, y) within the block is:

```
h_offset = round_div(delta_h * (2 * x - (W - 1)), 2 * (W - 1))
v_offset = round_div(delta_v * (2 * y - (H - 1)), 2 * (H - 1))
pred[x][y] = clamp(dc + h_offset + v_offset, 0, (1 << BitDepth) - 1)
```

The gradients are applied relative to the block centre. The exact integer arithmetic and intermediate value ranges are specified in Appendix A.

### 5.4 Inter Prediction: Motion Compensation

#### 5.4.1 Reference Index

Each inter-predicted block (mode 1) uses a single reference frame from the DPB.

If the DPB contains exactly 1 frame, the reference index is implicitly 0.

If the DPB contains N > 1 frames, the reference index is decoded as a symbol from an N-entry alphabet (maximum 8):

```
ref_idx = rans_decode(ref_idx_cdf[0])    // single context slot
```

DPB index 0 is the most recently decoded frame, index 1 is the next most recent, and so on.

For SKIP mode, the reference index is implicitly 0 (most recent frame).

#### 5.4.2 Motion Vector Coding

The decoder recovers a single motion vector with horizontal and vertical components at quarter-pel precision (int16 values in units of quarter-samples).

The MV is coded as a delta from the predictor (§5.4.3):

**Per component** (horizontal, then vertical):

**Delta class** (7-entry alphabet): Encodes the magnitude range.

| Class | Magnitude | Extra bits |
|-------|-----------|------------|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 2–3 | 1 |
| 3 | 4–7 | 2 |
| 4 | 8–15 | 3 |
| 5 | 16–31 | 4 |
| 6 | 32+ | Exp-Golomb order 0 of (value − 32) |

```
mvd_class_context = component    // 2 context slots (0=horizontal, 1=vertical)
```

For classes 2–5, extra bits are read from the bypass region (MSB first) to identify the exact value within the range. For class 6, the residual (value − 32) is coded as Exp-Golomb order 0 in bypass bits:

```
value = magnitude - 32
n = floor(log2(value + 1))
read n zero bits, then a 1 bit               // unary prefix
read n bits: value - (2^n - 1)                // binary suffix
```

**Sign** (bypass): One bit, present for non-zero deltas (0 = positive, 1 = negative).

The final MV component is `predictor_component + delta`.

#### 5.4.3 MV Predictor

The MV predictor is derived deterministically from spatial neighbours — no predictor index is coded.

Let (cx, cy) be the top-left cell of the current block. The left neighbour (A) is the block containing cell (cx − 1, cy). The above neighbour (B) is the block containing cell (cx, cy − 1). If a neighbour cell position is outside the tile boundary, or the neighbour block is intra-coded, that neighbour's MV is treated as (0, 0).

```
if A is available and B is available:
    mvp_x = (A.x + B.x) / 2                  // truncating division toward zero
    mvp_y = (A.y + B.y) / 2
else if A is available:
    mvp = A
else if B is available:
    mvp = B
else:
    mvp = (0, 0)
```

The truncating division toward zero is: `(a + b) / 2` where `/` truncates toward zero. The sum `A.x + B.x` shall be computed with int32 precision (the sum of two int16 values can range from −65536 to +65534). The normative implementation is:

```
trunc_avg(a, b):                     // a, b are int16
    sum = (int32)a + (int32)b        // int32 sum
    return (sum + ((sum >> 31) & 1)) >> 1
```

where `>> 31` extracts the sign bit of the 32-bit sum. This adds 1 before the right shift for negative odd sums, producing truncation toward zero rather than toward negative infinity.

**Note:** Using `>> 15` instead of `>> 31` produces incorrect results for sums with magnitude exceeding 32767 (e.g. when both MV components have the same sign and large magnitude).

#### 5.4.4 Integer Bilinear Interpolation

Motion compensation uses bilinear interpolation at quarter-pel precision. This is the simplest sub-pixel interpolation filter: four taps, four reference sample fetches, and four multiplies per output sample.

The rationale for bilinear interpolation (rather than the 6-tap or 8-tap filters used in other codecs) is threefold. First, it avoids patent encumbrance from specific filter coefficient designs. Second, it requires only four texture fetches per sample on a GPU (versus sixteen for bicubic), keeping the codec within the 2 Gtexel/s budget at 4K. Third, the CNN loop filter compensates for the quality difference — it learns to correct the characteristic blurring and aliasing that bilinear interpolation produces.

**Reference frame padding:** Before interpolation, reference samples outside the frame boundary are generated by clamping coordinates to the valid range:

```
ref_sample(ry, rx) = ref[clamp(ry, 0, ref_height-1)][clamp(rx, 0, ref_width-1)]
```

**Normative interpolation process:**

The motion vector for the current block is (mvx, mvy) in quarter-pel units. Decompose into integer and fractional parts:

```
ix = mvx >> 2            // arithmetic right shift (sign-extending)
iy = mvy >> 2
fx = mvx & 3             // fractional position (0–3)
fy = mvy & 3
```

For each output sample at block position (bx, by), compute the reference position:

```
rx = block_origin_x + bx + ix
ry = block_origin_y + by + iy
```

Fetch four reference samples (with padding):

```
s00 = ref_sample(ry,     rx    )
s10 = ref_sample(ry,     rx + 1)
s01 = ref_sample(ry + 1, rx    )
s11 = ref_sample(ry + 1, rx + 1)
```

Compute the interpolated value using int32 arithmetic:

```
h0 = s00 * (4 - fx) + s10 * fx
h1 = s01 * (4 - fx) + s11 * fx
pred = round_shift(h0 * (4 - fy) + h1 * fy, 4)
pred = clamp(pred, 0, (1 << BitDepth) - 1)
```

---

## 6. Transform and Quantisation

### 6.1 Design Rationale

Lattice uses integer approximations of the DCT-II transform, using the well-established Chen/Wang factorisation. Integer DCT approximations have extensive prior art predating modern codec patents.

Transform sizes are 4×4 (for chroma blocks derived from 8×8 luma blocks in 4:2:0), 8×8, 16×16, and 32×32. The transform size always matches the block size — there is no separate transform size signalling. This is a direct consequence of the power-of-two block shape constraint: every block dimension is a valid transform length.

No secondary transforms are used. Tools such as VVC's low-frequency non-separable transform (LFNST) are avoided for patent reasons. The CNN loop filter provides an alternative mechanism for recovering the small coding gain these tools offer.

Dequantisation applies a normative perceptual frequency weight to each coefficient position (§6.3.2). The weight increases quadratically with radial frequency, so high-frequency coefficients are quantised more coarsely — matching the well-established characteristic of the human contrast sensitivity function (CSF). Unlike HEVC or VVC, the weight is a fixed formula with no signalling: the CNN loop filter provides per-frame adaptation that would otherwise require signalled quantisation matrices.

### 6.2 Inverse Transform

The transform is separable: a 1D inverse DCT-II is applied along rows, followed by a 1D inverse DCT-II along columns. The 1D kernels are normative coefficient matrices for each size (4, 8, 16, 32 points), specified in Appendix B.

Input coefficients are int16 (dequantised values from §6.3.3). Intermediate arithmetic is int32. The 2D inverse transform is computed as two passes:

1. **Horizontal pass**: Apply the 1D inverse DCT to each row of the coefficient block. Right-shift each output by `TRANSFORM_SHIFT_1` = 7 and clamp to [−32768, 32767] (int16). The clamp prevents accumulator overflow in the vertical pass.

2. **Vertical pass**: Apply the 1D inverse DCT to each column of the intermediate block. Right-shift each output by `TRANSFORM_SHIFT_2` = 20 − BitDepth (12 for 8-bit, 10 for 10-bit) to produce int16 residual values.

The 1D inverse DCT of N input coefficients X[0..N−1] is:

```
for n in 0..N-1:
    acc = 0                          // int32
    for k in 0..N-1:
        acc += C[k][n] * X[k]       // C from Appendix B
    output[n] = acc
```

### 6.3 Quantisation

#### 6.3.1 QP Delta

The QP delta is a small signed integer coded per block (except SKIP blocks, which have delta_qp = 0). It is decoded as:

```
// Alphabet: 5 entries representing delta values -2, -1, 0, +1, +2
// Symbol 0 = delta -2, symbol 2 = delta 0, symbol 4 = delta +2
delta_qp = rans_decode(qp_delta_cdf[qp_delta_context]) - 2
```

```
qp_delta_context = (above_block.delta_qp != 0 ? 1 : 0)
                 + (left_block.delta_qp != 0 ? 1 : 0)
// qp_delta_context: 0, 1, or 2
```

If the above or left neighbour is unavailable (block at a tile edge or at the first row/column of the tile), that neighbour contributes 0 to the context sum (equivalent to `delta_qp = 0`).

The block QP is:

```
block_qp = clamp(base_qp + delta_qp, 0, 51)
```

#### 6.3.2 Perceptual Frequency Weight

Each coefficient position (i, j) in the transform block carries a normative weight that models the human visual system's decreasing sensitivity to high spatial frequencies. Position (0, 0) is DC; higher indices correspond to higher frequencies. The weight is:

```
PW[i][j] = min(16 + i*i + j*j, 112)
```

where i is the row (vertical frequency index) and j is the column (horizontal frequency index), both zero-based. The constant 16 is the neutral baseline (it will cancel in the dequantisation scale). The cap at 112 gives a maximum quantisation coarsening ratio of 112/16 = 7:1 relative to DC.

**Rationale.** DCT coefficient (i, j) has radial spatial frequency proportional to √(i² + j²). The quadratic form i² + j² is thus proportional to the squared spatial frequency. Perceptual sensitivity (the inverse of the detection threshold) falls off steeply with spatial frequency; the quadratic weight approximates this falloff. The position-only formula naturally adapts across transform sizes: a 4×4 chroma block reaches at most PW[3][3] = 34 (modest 2.1× shaping), while a 32×32 luma block saturates most of its high-frequency area at 112 (aggressive 7× shaping). No signalling is required.

The weight table may be precomputed at decoder initialisation. The maximum table size is 32 × 32 = 1024 bytes. Reference tables for 4×4 and 8×8 are given in Appendix F.

#### 6.3.3 Dequantisation

The quantisation step size `qstep` is looked up from the block QP using the table in Appendix E. Each dequantised transform coefficient is computed from the coded level (the absolute value recovered from coefficient coding, §9) and the perceptual frequency weight (§6.3.2):

```
effective_qstep = (qstep * PW[i][j] + 8) >> 4
dequant_coeff   = level * effective_qstep
```

where `level` is int16 (signed, from coefficient coding), `qstep` is uint16, and PW[i][j] is uint8 in [16, 112].

At position (0, 0): PW = 16, so `effective_qstep = (qstep * 16 + 8) >> 4 = qstep` — DC is quantised at the base step size. At the highest frequencies: PW = 112, so `effective_qstep ≈ 7 × qstep` — those coefficients tolerate ~7× coarser quantisation.

The product `qstep * PW[i][j]` is at most 9216 × 112 = 1,032,192, which fits in uint32. After the shift, `effective_qstep` is at most (1,032,192 + 8) >> 4 = 64,512, which fits in uint16.

**Note on bit width:** For the highest QP (51, effective_qstep = 64,512 at corner positions) and maximum coded level, the maximum dequantised coefficient magnitude is 32767 × 64,512 ≈ 2.11 × 10⁹, which fits in int32 (max 2.15 × 10⁹). The inverse transform accumulator must handle the sum of N such values multiplied by transform coefficients. The horizontal pass shift of 7 and the int16 clamp between passes (§6.2) prevent overflow.

### 6.4 Reconstruction

For blocks with CBF = 1:

```
recon[x][y] = clamp(pred[x][y] + residual[x][y], 0, (1 << BitDepth) - 1)
```

For blocks with CBF = 0 (including SKIP):

```
recon[x][y] = pred[x][y]
```

---

## 7. Entropy Coding

### 7.1 Design Rationale

Lattice uses rANS (range Asymmetric Numeral Systems) rather than arithmetic coding (CABAC). The choice is motivated by three factors:

**Patent safety.** ANS was published openly by Jarek Duda and is generally considered unencumbered. CABAC's specific implementation details in H.264/H.265/H.266 are covered by numerous patents.

**Multi-symbol efficiency.** rANS natively handles multi-symbol alphabets without binarisation. A single rANS decode operation recovers a symbol from an alphabet of up to 16 entries, whereas CABAC must binarise every symbol and process each bin serially. This dramatically reduces the number of serial decode steps per block.

**Simplicity.** The rANS decode loop is approximately 15 lines of C. The core operation is a shift, a mask, a table lookup, and a multiply — no division. On a Cortex-A76, each decode step takes approximately 8–12 cycles.

### 7.2 Core rANS Engine

The rANS decoder maintains a 32-bit unsigned state `x` in the interval [2^16, 2^32). CDF tables use M = 2^16 = 65536 as the probability resolution.

To decode one symbol from a CDF table `cdf[]` of size N+1 (representing N symbols, with `cdf[0] = 0` and `cdf[N] = M`):

```
q = x >> 16                              // upper 16 bits
r = x & 0xFFFF                           // lower 16 bits
s = lookup(cdf, r)                       // find s where cdf[s] <= r < cdf[s+1]
freq = cdf[s+1] - cdf[s]
x = q * freq + (r - cdf[s])             // state update

while (x < (1 << 16)):                  // renormalisation
    x = (x << 8) | read_byte()          // read from appropriate stream
```

The `lookup` function finds the symbol whose cumulative range contains `r`. For alphabets of size ≤ 16, a linear scan from `s = 0` upward, stopping when `cdf[s+1] > r`, is both correct and branch-predictor-friendly.

No division is required anywhere in this process. The multiply is 32×16→32 (the `freq` value is at most 16 bits by construction).

### 7.3 Dual-Stream Interleaving

Each tile contains two independent rANS streams. Context-coded symbols are assigned to streams in strict alternation: the first context-coded symbol in the tile goes to stream 0, the second to stream 1, the third to stream 0, and so on. The decoder maintains a single-bit toggle, starting at 0, which selects the active stream.

Stream 0 reads bytes forward. Stream 1 reads bytes backward. Both maintain independent 32-bit states initialised as specified in §3.5.

The rationale for dual-stream interleaving is throughput. Each rANS decode step has a serial data dependency on the previous step's state update. By alternating between two independent state machines, the decoder overlaps the multiply latency from one stream with the CDF lookup from the other. On an out-of-order CPU this roughly doubles throughput. In the hardware decoder, two rANS units run in lockstep.

The backward-reading direction of stream 1 is a natural consequence of ANS encoding being a LIFO process. The encoder writes stream 1's bytes in production order, and the decoder reads them in reverse. This allows both streams to be packed without padding — they grow toward each other from opposite ends of the rANS region.

Both streams' states are reset at tile boundaries. There is no cross-tile entropy state.

### 7.4 CDF Adaptation

Each context slot stores a CDF over a small alphabet (sizes given in §7.6). After decoding symbol `s` from a context with alphabet size N, the CDF is updated:

```
for i in 1..N-1:
    if i <= s:
        cdf[i] += (0 - cdf[i]) >> ADAPT_RATE
    else:
        cdf[i] += (65536 - cdf[i]) >> ADAPT_RATE
```

where `ADAPT_RATE` = 5 (adaptation rate 1/32, half-life approximately 22 symbols). The update shifts each CDF boundary toward the observed symbol, with the boundaries below the symbol moving toward 0 and boundaries above moving toward 65536.

#### 7.4.1 Minimum Frequency Enforcement

After adaptation, the decoder enforces that every symbol has at least frequency 1 (i.e., `cdf[i+1] − cdf[i] ≥ 1` for all i). The enforcement algorithm is:

```
// After adaptation of a CDF with N symbols:
for i in 0..N-2:
    min_value = cdf[i] + 1                     // must be > previous entry
    max_value = cdf[N] - (N - 1 - i)           // must leave room for remaining symbols
    cdf[i + 1] = clamp(cdf[i + 1], min_value, max_value)
```

This ensures that `cdf` remains a valid, strictly increasing sequence from `cdf[0] = 0` to `cdf[N] = 65536`, with each bin having frequency ≥ 1.

#### 7.4.2 CDF Initialisation

All CDFs are initialised to a uniform distribution at the start of each tile:

```
for i in 0..N:
    cdf[i] = (65536 * i) / N                  // integer division
```

### 7.5 Bypass Bits

Sign bits, MV extra bits, and coefficient escape suffixes are coded as raw bits in the bypass region (§3.4). The bypass reader is specified in §2.4.

The bitstream syntax deterministically specifies when the decoder reads from the rANS streams versus the bypass region. There is no explicit signalling for the switch — it is implicit in the syntax element being decoded (e.g., "read sign bit from bypass" versus "decode level token from rANS").

### 7.6 Context Allocation

The following table gives the complete context allocation. Context indices are assigned sequentially starting from 0.

| Category | Base Index | Count | Alphabet | Total Slots |
|----------|-----------|-------|----------|-------------|
| Block map (shape_index) | 0 | 9 | ≤7 | 9 |
| Prediction mode | 9 | 9 | 3 | 9 |
| CBF | 18 | 3 | 2 | 3 |
| QP delta | 21 | 3 | 5 | 3 |
| Reference index | 24 | 1 | ≤8 | 1 |
| MV delta class | 25 | 2 | 7 | 2 |
| Coeff band status (luma) | 27 | 8 | 2 | 8 |
| Coeff significance (luma) | 35 | 16 | 2 | 16 |
| Coeff level tokens (luma) | 51 | 16 | 8 | 16 |
| Coeff band status (chroma) | 67 | 8 | 2 | 8 |
| Coeff significance (chroma) | 75 | 16 | 2 | 16 |
| Coeff level tokens (chroma) | 91 | 16 | 8 | 16 |
| Filter weight delta | 107 | 3 | 9 | 3 |
| **Total** | | | | **110** |

The band status category has 8 slots per plane: `band_index * 2 + prev_zero` produces indices 0–7 for 4 bands (§9.4). The significance and level categories each use 16 slots: `min(band_index, 3) * 4 + sub_context` produces indices 0–15.

Each context slot stores a CDF array of `alphabet_size + 1` uint16 values. The maximum CDF size is 10 entries (for the 9-symbol filter weight delta alphabet). At 2 bytes per entry, the total context memory is at most 110 × 10 × 2 = 2,200 bytes.

---

## 8. Loop Filter

### 8.1 Design Rationale

Traditional codecs apply a cascade of post-reconstruction filters: deblocking to smooth block boundaries, sample-adaptive offset (SAO) to correct systematic biases, and adaptive loop filtering (ALF) to minimise reconstruction error. Each filter has its own syntax, signalling overhead, and decoder implementation.

Lattice replaces this entire cascade with a single convolutional neural network applied to each tile after reconstruction. The CNN's architecture is fixed and normative; its weights are either baked-in defaults or signalled per frame. This single tool subsumes the functions of deblocking, SAO, and ALF: it learns to smooth block boundaries, correct per-sample biases, and apply spatially adaptive filtering, all within a unified framework.

The CNN uses 4 feature channels. This width is constrained by the compute budgets of the target implementations: at 4 channels, the filter requires approximately 360 MACs per output sample, which is achievable at 4K30 on the hardware MAC array, at 1080p30 on the mobile GPU, and at 1080p30 on two A76 cores.

### 8.2 Filter Architecture

The CNN has 4 convolutional layers, each using 3×3 kernels with ReLU activation (except the output of the final layer):

| Layer | Input Channels | Output Channels | Weights | Biases | Total Parameters |
|-------|---------------|----------------|---------|--------|-----------------|
| 1 | 1 | 4 | 36 | 4 | 40 |
| 2 | 4 | 4 | 144 | 4 | 148 |
| 3 | 4 | 4 | 144 | 4 | 148 |
| 4 | 4 | 1 | 36 | 1 | 37 |
| **Total** | | | | | **373** |

Layer 1 takes the reconstructed sample plane as single-channel input and expands to 4 feature channels. Layers 2 and 3 process the 4-channel feature maps. Layer 4 contracts back to a single-channel filtered output.

All convolutions use same-padding: the input is extended at borders by replicating edge samples. The spatial dimensions are preserved through all layers.

### 8.3 Normative Computation

The output of each layer, for each output position (x, y) and output channel c_out, is computed as:

```
acc = bias[c_out]                                  // int32
for c_in in 0..channels_in - 1:                    // normative order
    for ky in 0..2:                                // normative order
        for kx in 0..2:                            // normative order
            // (x+kx-1, y+ky-1) with replicate-edge padding
            sx = clamp(x + kx - 1, 0, plane_width - 1)
            sy = clamp(y + ky - 1, 0, plane_height - 1)
            acc += weight[c_out][c_in][ky][kx] * activation[c_in][sy][sx]
acc = round_shift(acc, FILTER_SHIFT)
```

For layers 1–3: `output[c_out][y][x] = clamp(acc, 0, 2047)` (ReLU + 11-bit clamp).

For layer 4: `output[0][y][x] = clamp(acc, 0, (1 << BitDepth) - 1)` (final sample output).

The accumulation order — input channel, then kernel row, then kernel column — is normative.

### 8.4 Fixed-Point Arithmetic

- **Weights**: int12 signed (range −2048 to +2047)
- **Activations**: uint11 for layers 2–4 input (range 0–2047); layer 1 input is uint8 or uint10 (sample values), zero-extended to uint11
- **Bias**: int32 (stored as int12 signed, sign-extended to int32 before use)
- **Accumulator**: int32
- **FILTER_SHIFT**: 10 (normative constant)

Worst-case accumulation for layers 2–3 (4 input channels × 9 positions = 36 MACs): 36 × 2048 × 2047 = 150,994,944, well within int32 range.

### 8.5 Weight Signalling

When `filter_mode` = 0: default weights (Appendix D) are used for both luma and chroma.

When `filter_mode` = 1: custom luma weights follow in the frame header. Chroma always uses default weights.

The rationale for not signalling custom chroma weights is that chroma planes are half-resolution and perceptually less sensitive. The identity defaults pass chroma through unfiltered, which the CNN filter on the luma plane cannot affect — but chroma artefacts at half resolution are less objectionable, and the signalling cost is not justified.

Custom luma weights are coded as 373 delta values against the defaults. Filter weight deltas are coded using a single forward-reading rANS stream, framed by `filter_rans_size` in the frame header (§3.3). The rANS state is initialised by reading the first 4 bytes of `filter_rans_data` in big-endian order; subsequent bytes are read forward. Dual-stream interleaving (§7.3) applies only to tile data, not to filter weight coding. The three filter weight context CDFs (§7.6) are initialised to uniform at the start of filter weight decoding.

Each delta is coded as a symbol from a 9-entry alphabet representing values −4 through +4:

```
for each parameter p[i] in order (layer 1 weights, layer 1 biases,
                                   layer 2 weights, layer 2 biases,
                                   layer 3 weights, layer 3 biases,
                                   layer 4 weights, layer 4 biases):
    delta_symbol = rans_decode(filter_delta_cdf[filter_delta_context])
    delta = delta_symbol - 4                   // range -4 to +4
    p[i] = clamp(default_p[i] + delta, -2048, 2047)
```

The context for filter weight deltas cycles through 3 slots based on `i % 3`. After decoding each symbol, the corresponding context CDF is adapted using the standard procedure (§7.4).

### 8.6 Processing Order

The loop filter is applied to each tile after all blocks within that tile have been fully reconstructed. The input is the tile's reconstructed sample values. At tile borders, the input is padded:

- At borders adjacent to other tiles in the same frame: reconstructed (pre-filter) samples from the adjacent tile are used.
- At frame boundaries: replicate-edge padding.

The filter is applied independently to luma and chroma planes, using their respective weight sets.

The filter output replaces the reconstructed samples in the decoded picture buffer. These filtered samples are used as reference for subsequent inter-predicted frames.

**Normative processing model.** When filtering tile T, the CNN reads pre-filter (reconstructed, not yet filtered) samples from all adjacent tiles for its border padding. This defines the normative output: every conformant decoder must produce the same filtered sample values regardless of implementation strategy. Two implementation approaches are conformant:

1. **Frame-at-a-time:** Reconstruct all tiles, storing pre-filter samples in a buffer. Then filter every tile in a second pass, reading neighbour padding from the pre-filter buffer.

2. **Pipelined:** Filter each tile as soon as all of its adjacent tiles (including diagonals, since the 3×3 kernel reaches corner neighbours) have been reconstructed. With raster-order tile reconstruction, tile T at row r, column c can be filtered once tile (r+1, c+1) has been reconstructed — a lag of approximately one tile row plus one tile. Pre-filter samples from adjacent tiles must be preserved until all tiles that reference them have been filtered.

**Clarification:** The filter for tile T shall never read *filtered* output from another tile. All cross-tile padding uses *pre-filter* reconstructed samples. This ensures that tile filtering order does not affect the decoded output.

---

## 9. Coefficient Coding

### 9.1 Overview

Transform coefficients typically comprise 60–80% of the coded bits in a Lattice bitstream. This section specifies the band-grouped coding scheme used to code them efficiently while minimising serial decode steps.

Coefficients are coded separately for luma and chroma. For a block with CBF = 1, the luma coefficients are coded first, then Cb, then Cr. Each plane's coding process is identical in structure but uses independent context model states.

### 9.2 Scan Order

Coefficients within each transform block are ordered by a fixed diagonal scan, proceeding from DC (top-left) to the highest frequency (bottom-right). The scan follows anti-diagonals: position 0 is (0,0), position 1 is (1,0), position 2 is (0,1), and so on.

The scan algorithm and tables are specified in Appendix C.

### 9.3 Frequency Bands

Each transform block is divided into frequency bands based on scan position. The band boundaries are defined by anti-diagonal index f = u + v.

For blocks with both dimensions ≥ 8:

| Band | Anti-diagonals | Scan positions |
|------|---------------|----------------|
| 0 | f = 0 (DC only) | pos 0 |
| 1 | f = 1, 2 | pos 1–5 |
| 2 | f = 3, 4, 5, 6 | pos 6–27 |
| 3 | f ≥ 7 | pos 28 to N−1 |

For 4×4 blocks (3 bands only):

| Band | Anti-diagonals | Scan positions |
|------|---------------|----------------|
| 0 | f = 0 | pos 0 |
| 1 | f = 1, 2 | pos 1–5 |
| 2 | f ≥ 3 | pos 6–15 |

### 9.4 Per-Band Coding Process

Bands are coded in order from band 0 to the highest band. For each band:

**Step 1 — Band status.** A binary flag:

| Value | Name | Meaning |
|-------|------|---------|
| 0 | `ALL_ZERO` | All coefficients in this band are zero. Skip to next band. |
| 1 | `CODED` | Significance flags and levels follow. |

Context:
```
prev_zero = (band_index > 0 && previous_band_status == ALL_ZERO) ? 1 : 0
band_status_context = BAND_STATUS_BASE + band_index * 2 + prev_zero
```

**Step 2 — Significance flags** (CODED bands only). One binary flag per coefficient position in the band, in scan order. 1 = non-zero, 0 = zero.

```
// density = count of non-zero flags in last 4 decoded flags, saturated to 3
sig_context = SIG_BASE + min(band_index, 3) * 4 + density
```

**Step 3 — Level tokens.** For each non-zero coefficient (flagged positions from step 2), the absolute level is decoded:

```
level_token = rans_decode(level_cdf[level_context])    // 0..7
```

Tokens 0–6 represent absolute levels 1–7 directly. Token 7 is ESCAPE, meaning the level is 8 or greater.

```
prev_level_cat = 0 if prev_level <= 1, 1 if prev_level <= 4, 2 if prev_level <= 7, 3 otherwise
level_context = LEVEL_BASE + min(band_index, 3) * 4 + prev_level_cat
```

`prev_level` is the absolute level of the most recently decoded non-zero coefficient in this block (0 at block start).

**Step 4 — Escape suffix.** If the level token is ESCAPE (7), the value `level − 8` is decoded as Exp-Golomb order 0 in bypass bits:

```
value = 0
n = 0
while (read_bypass_bit() == 0):
    n += 1
for i in n-1 downto 0:
    value = (value << 1) | read_bypass_bit()
value += (1 << n) - 1
absolute_level = value + 8
```

**Step 5 — Signs.** One bypass bit per non-zero coefficient, in scan order (0 = positive, 1 = negative).

### 9.5 Luma and Chroma Context Separation

Luma coefficients use context slots starting at `BAND_STATUS_BASE(luma)`, `SIG_BASE(luma)`, `LEVEL_BASE(luma)`. Chroma uses separate slots at offset base indices as specified in §7.6. Cb and Cr share the same chroma context states (they are decoded sequentially and the shared adaptation benefits both planes).

---

## 10. Chroma

### 10.1 Colour Format

Lattice supports 4:2:0 chroma subsampling. Each chroma plane (Cb, Cr) has half the width and half the height of the luma plane, computed using ceiling division:

```
chroma_width  = (frame_width + 1) / 2       // ceiling division
chroma_height = (frame_height + 1) / 2
```

These dimensions define the chroma plane sample grid, the chroma sample buffer allocation, and the output format. For odd luma dimensions, the last chroma row or column corresponds to the last 1 luma row or column (rather than the usual 2). Implementations shall allocate chroma buffers of exactly `chroma_width × chroma_height` samples.

### 10.2 Chroma Block Derivation

Chroma blocks are derived directly from the luma block map by halving all coordinates and dimensions. A luma block of size W×H at position (x, y) produces a chroma block of size W/2 × H/2 at position (x/2, y/2).

The minimum luma block is 8×8, producing a minimum chroma block of 4×4. This requires a 4-point inverse DCT kernel (Appendix B).

### 10.3 Chroma Prediction

Chroma prediction uses the same mode as luma:

- `INTRA`: The gradient prediction process (§5.3) is applied independently to the chroma plane using its own reconstructed neighbours.
- `INTER`: The luma MV is halved (each component divided by 2 with truncation toward zero) to produce the chroma MV. The same reference frame is used. The bilinear interpolation process (§5.4.4) is applied at the chroma-domain MV.
- `SKIP`: Same as INTER, using the predictor MV.

### 10.4 Chroma QP

Chroma uses the same `block_qp` as luma. The `qstep` value is the same. The perceptual frequency weight (§6.3.2) is applied using the chroma block's own coefficient positions — since chroma blocks are half the luma size (minimum 4×4), the weight naturally reaches lower maxima (e.g. PW[3][3] = 34 for 4×4), giving gentler perceptual shaping appropriate for the subsampled chroma planes.

---

## 11. Decoded Picture Buffer

The decoder maintains a decoded picture buffer (DPB) of up to `max_ref_frames` decoded frames (signalled in the sequence header, maximum 8). Each DPB entry stores the filtered luma, Cb, and Cr planes.

After a frame is decoded and filtered:
1. If the DPB contains `max_ref_frames` entries, the oldest entry is evicted.
2. The newly decoded and filtered frame is inserted as the newest entry.

DPB entries are indexed 0 to `dpb_count − 1`, where `dpb_count` is the current number of frames in the DPB. Index 0 is the most recently inserted frame; higher indices are progressively older frames.

The decoder tracks `dpb_count` internally. It starts at 0 and increases by 1 after each decoded frame, up to `max_ref_frames`. On inter frames, all `dpb_count` entries are available as references. The per-block `ref_idx` (§5.4.1) selects among them.

On intra frames (`frame_type` = 0), no references are used. The DPB is **not** flushed — previously decoded frames remain available for subsequent inter frames.

---

## 12. Decoding Process Summary

The complete decoding process for one frame is:

```
1. Read frame header (§3.3)
2. If filter_mode = 1, decode custom luma filter weights (§8.5)
3. For each tile in raster order (left to right, top to bottom):
   a. Read tile header (§3.4)
   b. Initialise both rANS streams (§3.5)
   c. Initialise all CDF contexts to uniform (§7.4.2)
   d. Initialise bypass bit reader at bypass_offset
   e. Decode block map (§4.3)
   f. For each block in raster order (§5.1):
      i.   Decode prediction mode (if inter frame)
      ii.  If INTER and dpb_count > 1: decode ref_idx
      iii. If INTER: decode MV delta
      iv.  If not SKIP: decode QP delta
      v.   If not SKIP: decode CBF
      vi.  If CBF=1: decode coefficients (luma, Cb, Cr)
      vii. Generate prediction (intra or inter)
      viii.If CBF=1: apply perceptual weight (§6.3.2), dequantise (§6.3.3),
                     inverse transform, add residual
      ix.  Reconstruct: clamp to valid sample range
   g. Apply CNN loop filter to luma plane (§8)
   h. Apply CNN loop filter to Cb plane (using default weights)
   i. Apply CNN loop filter to Cr plane (using default weights)
4. Insert filtered frame into DPB (§11)
5. Output decoded frame
```

---

## Appendix A: Intra Prediction Fixed-Point Arithmetic

### A.1 Integer Division

The intra prediction derivation (§5.3.1) requires division by small constants. The normative definition is:

```
round_div(a, d):
    if a >= 0:
        return (a + (d >> 1)) // d
    else:
        return -((-a + (d >> 1)) // d)
```

where `//` denotes truncating integer division and `d` is always a positive integer. This computes the nearest integer to a/d, with ties rounding away from zero.

The divisors that appear in intra prediction are:

| Context | Divisor values |
|---------|---------------|
| DC (sum / count) | W, H ∈ {8, 16, 32}; W+H ∈ {16, 24, 32, 48, 64} |
| Gradient scaling | 2·(W−1), 2·(H−1) ∈ {14, 30, 62} |

**Informative note:** Hardware implementations may replace `round_div(a, d)` with `(a * M + (1 << (S-1))) >> S` using the following table (exact for |a| ≤ 65536):

| Divisor | Multiplier M | Shift S |
|---------|-------------|---------|
| 8 | 65537 | 19 |
| 14 | 74899 | 20 |
| 16 | 65537 | 20 |
| 24 | 87382 | 21 |
| 30 | 69906 | 21 |
| 32 | 65537 | 21 |
| 48 | 87382 | 22 |
| 62 | 67651 | 22 |
| 64 | 65537 | 22 |

### A.2 Gradient Prediction Arithmetic

The intermediate value `delta_h * (2·x − (W−1))` has magnitude at most 1023 × 62 = 63,426. The result of `round_div` has magnitude at most 63,426 / 14 = 4,531. Both fit in int16. The final sum `dc + h_offset + v_offset` may exceed the sample range and is clamped.

---

## Appendix B: Inverse Transform Kernels

### B.1 Generating Formula

The N-point integer DCT-II coefficient matrix C[k][n] is defined by:

```python
import math

def generate_dct_matrix(N):
    C = []
    s0 = 64.0
    sk = 64.0 * math.sqrt(2.0)
    for k in range(N):
        row = []
        s = s0 if k == 0 else sk
        for n in range(N):
            val = s * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
            row.append(round(val))
        C.append(row)
    return C
```

When executed under CPython 3.10 or later with IEEE 754 binary64 floating-point, this function produces the normative coefficient tables below. In the event of a discrepancy between this code and the printed tables, re-running the code is definitive.

### B.2 4-Point Coefficient Matrix

```
C4[k][n], k=0..3, n=0..3:

    [  64,   64,   64,   64]
    [  84,   35,  -35,  -84]
    [  64,  -64,  -64,   64]
    [  35,  -84,   84,  -35]
```

### B.3 8-Point Coefficient Matrix

```
C8[k][n], k=0..7, n=0..7:

    [  64,   64,   64,   64,   64,   64,   64,   64]
    [  89,   75,   50,   18,  -18,  -50,  -75,  -89]
    [  84,   35,  -35,  -84,  -84,  -35,   35,   84]
    [  75,  -18,  -89,  -50,   50,   89,   18,  -75]
    [  64,  -64,  -64,   64,   64,  -64,  -64,   64]
    [  50,  -89,   18,   75,  -75,  -18,   89,  -50]
    [  35,  -84,   84,  -35,  -35,   84,  -84,   35]
    [  18,  -50,   75,  -89,   89,  -75,   50,  -18]
```

### B.4 16-Point Coefficient Matrix

```
C16[k][n], k=0..15, n=0..15:

    [  64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64]
    [  90,   87,   80,   70,   57,   43,   26,    9,   -9,  -26,  -43,  -57,  -70,  -80,  -87,  -90]
    [  89,   75,   50,   18,  -18,  -50,  -75,  -89,  -89,  -75,  -50,  -18,   18,   50,   75,   89]
    [  87,   57,    9,  -43,  -80,  -90,  -70,  -26,   26,   70,   90,   80,   43,   -9,  -57,  -87]
    [  84,   35,  -35,  -84,  -84,  -35,   35,   84,   84,   35,  -35,  -84,  -84,  -35,   35,   84]
    [  80,    9,  -70,  -87,  -26,   57,   90,   43,  -43,  -90,  -57,   26,   87,   70,   -9,  -80]
    [  75,  -18,  -89,  -50,   50,   89,   18,  -75,  -75,   18,   89,   50,  -50,  -89,  -18,   75]
    [  70,  -43,  -87,    9,   90,   26,  -80,  -57,   57,   80,  -26,  -90,   -9,   87,   43,  -70]
    [  64,  -64,  -64,   64,   64,  -64,  -64,   64,   64,  -64,  -64,   64,   64,  -64,  -64,   64]
    [  57,  -80,  -26,   90,   -9,  -87,   43,   70,  -70,  -43,   87,    9,  -90,   26,   80,  -57]
    [  50,  -89,   18,   75,  -75,  -18,   89,  -50,  -50,   89,  -18,  -75,   75,   18,  -89,   50]
    [  43,  -90,   57,   26,  -87,   70,    9,  -80,   80,   -9,  -70,   87,  -26,  -57,   90,  -43]
    [  35,  -84,   84,  -35,  -35,   84,  -84,   35,   35,  -84,   84,  -35,  -35,   84,  -84,   35]
    [  26,  -70,   90,  -80,   43,    9,  -57,   87,  -87,   57,   -9,  -43,   80,  -90,   70,  -26]
    [  18,  -50,   75,  -89,   89,  -75,   50,  -18,  -18,   50,  -75,   89,  -89,   75,  -50,   18]
    [   9,  -26,   43,  -57,   70,  -80,   87,  -90,   90,  -87,   80,  -70,   57,  -43,   26,   -9]
```

### B.5 32-Point Coefficient Matrix

The 32-point matrix is generated by `generate_dct_matrix(32)` as defined in §B.1.

```
C32[k][n], k=0..31, n=0..31:

k= 0: [  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64]
k= 1: [  90,  90,  88,  85,  82,  78,  73,  67,  61,  54,  47,  39,  30,  22,  13,   4,  -4, -13, -22, -30, -39, -47, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90]
k= 2: [  90,  87,  80,  70,  57,  43,  26,   9,  -9, -26, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -26,  -9,   9,  26,  43,  57,  70,  80,  87,  90]
k= 3: [  90,  82,  67,  47,  22,  -4, -30, -54, -73, -85, -90, -88, -78, -61, -39, -13,  13,  39,  61,  78,  88,  90,  85,  73,  54,  30,   4, -22, -47, -67, -82, -90]
k= 4: [  89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,  89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89]
k= 5: [  88,  67,  30, -13, -54, -82, -90, -78, -47,  -4,  39,  73,  90,  85,  61,  22, -22, -61, -85, -90, -73, -39,   4,  47,  78,  90,  82,  54,  13, -30, -67, -88]
k= 6: [  87,  57,   9, -43, -80, -90, -70, -26,  26,  70,  90,  80,  43,  -9, -57, -87, -87, -57,  -9,  43,  80,  90,  70,  26, -26, -70, -90, -80, -43,   9,  57,  87]
k= 7: [  85,  47, -13, -67, -90, -73, -22,  39,  82,  88,  54,  -4, -61, -90, -78, -30,  30,  78,  90,  61,   4, -54, -88, -82, -39,  22,  73,  90,  67,  13, -47, -85]
k= 8: [  84,  35, -35, -84, -84, -35,  35,  84,  84,  35, -35, -84, -84, -35,  35,  84,  84,  35, -35, -84, -84, -35,  35,  84,  84,  35, -35, -84, -84, -35,  35,  84]
k= 9: [  82,  22, -54, -90, -61,  13,  78,  85,  30, -47, -90, -67,   4,  73,  88,  39, -39, -88, -73,  -4,  67,  90,  47, -30, -85, -78, -13,  61,  90,  54, -22, -82]
k=10: [  80,   9, -70, -87, -26,  57,  90,  43, -43, -90, -57,  26,  87,  70,  -9, -80, -80,  -9,  70,  87,  26, -57, -90, -43,  43,  90,  57, -26, -87, -70,   9,  80]
k=11: [  78,  -4, -82, -73,  13,  85,  67, -22, -88, -61,  30,  90,  54, -39, -90, -47,  47,  90,  39, -54, -90, -30,  61,  88,  22, -67, -85, -13,  73,  82,   4, -78]
k=12: [  75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,  75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75]
k=13: [  73, -30, -90, -22,  78,  67, -39, -90, -13,  82,  61, -47, -88,  -4,  85,  54, -54, -85,   4,  88,  47, -61, -82,  13,  90,  39, -67, -78,  22,  90,  30, -73]
k=14: [  70, -43, -87,   9,  90,  26, -80, -57,  57,  80, -26, -90,  -9,  87,  43, -70, -70,  43,  87,  -9, -90, -26,  80,  57, -57, -80,  26,  90,   9, -87, -43,  70]
k=15: [  67, -54, -78,  39,  85, -22, -90,   4,  90,  13, -88, -30,  82,  47, -73, -61,  61,  73, -47, -82,  30,  88, -13, -90,  -4,  90,  22, -85, -39,  78,  54, -67]
k=16: [  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64]
k=17: [  61, -73, -47,  82,  30, -88, -13,  90,  -4, -90,  22,  85, -39, -78,  54,  67, -67, -54,  78,  39, -85, -22,  90,   4, -90,  13,  88, -30, -82,  47,  73, -61]
k=18: [  57, -80, -26,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  26,  80, -57, -57,  80,  26, -90,   9,  87, -43, -70,  70,  43, -87,  -9,  90, -26, -80,  57]
k=19: [  54, -85,  -4,  88, -47, -61,  82,  13, -90,  39,  67, -78, -22,  90, -30, -73,  73,  30, -90,  22,  78, -67, -39,  90, -13, -82,  61,  47, -88,   4,  85, -54]
k=20: [  50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,  50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50]
k=21: [  47, -90,  39,  54, -90,  30,  61, -88,  22,  67, -85,  13,  73, -82,   4,  78, -78,  -4,  82, -73, -13,  85, -67, -22,  88, -61, -30,  90, -54, -39,  90, -47]
k=22: [  43, -90,  57,  26, -87,  70,   9, -80,  80,  -9, -70,  87, -26, -57,  90, -43, -43,  90, -57, -26,  87, -70,  -9,  80, -80,   9,  70, -87,  26,  57, -90,  43]
k=23: [  39, -88,  73,  -4, -67,  90, -47, -30,  85, -78,  13,  61, -90,  54,  22, -82,  82, -22, -54,  90, -61, -13,  78, -85,  30,  47, -90,  67,   4, -73,  88, -39]
k=24: [  35, -84,  84, -35, -35,  84, -84,  35,  35, -84,  84, -35, -35,  84, -84,  35,  35, -84,  84, -35, -35,  84, -84,  35,  35, -84,  84, -35, -35,  84, -84,  35]
k=25: [  30, -78,  90, -61,   4,  54, -88,  82, -39, -22,  73, -90,  67, -13, -47,  85, -85,  47,  13, -67,  90, -73,  22,  39, -82,  88, -54,  -4,  61, -90,  78, -30]
k=26: [  26, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -26, -26,  70, -90,  80, -43,  -9,  57, -87,  87, -57,   9,  43, -80,  90, -70,  26]
k=27: [  22, -61,  85, -90,  73, -39,  -4,  47, -78,  90, -82,  54, -13, -30,  67, -88,  88, -67,  30,  13, -54,  82, -90,  78, -47,   4,  39, -73,  90, -85,  61, -22]
k=28: [  18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,  18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18]
k=29: [  13, -39,  61, -78,  88, -90,  85, -73,  54, -30,   4,  22, -47,  67, -82,  90, -90,  82, -67,  47, -22,  -4,  30, -54,  73, -85,  90, -88,  78, -61,  39, -13]
k=30: [   9, -26,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  26,  -9,  -9,  26, -43,  57, -70,  80, -87,  90, -90,  87, -80,  70, -57,  43, -26,   9]
k=31: [   4, -13,  22, -30,  39, -47,  54, -61,  67, -73,  78, -82,  85, -88,  90, -90,  90, -90,  88, -85,  82, -78,  73, -67,  61, -54,  47, -39,  30, -22,  13,  -4]
```

### B.6 2D Inverse Transform Summary

```
// Horizontal pass (each row):
for row in 0..H-1:
    for n in 0..W-1:
        acc = 0                              // int32
        for k in 0..W-1:
            acc += C_W[k][n] * coeff[row][k]
        intermediate[row][n] = clamp(round_shift(acc, 7), -32768, 32767)

// Vertical pass (each column):
for col in 0..W-1:
    for m in 0..H-1:
        acc = 0                              // int32
        for k in 0..H-1:
            acc += C_H[k][m] * intermediate[k][col]
        residual[m][col] = round_shift(acc, 20 - BitDepth)
```

---

## Appendix C: Diagonal Scan Order

### C.1 Generating Algorithm

```python
def diagonal_scan(W, H):
    """Generate diagonal scan for a W x H block.
    W = number of columns (horizontal frequencies).
    H = number of rows (vertical frequencies).
    Returns list of (u, v) tuples where u=column, v=row."""
    positions = []
    for f in range(W + H - 1):
        v_min = max(0, f - W + 1)
        v_max = min(f, H - 1)
        for v in range(v_min, v_max + 1):
            u = f - v
            positions.append((u, v))
    return positions
```

### C.2 Frequency Band Boundaries

```python
def band_boundaries(W, H):
    """Compute band start positions in scan order."""
    N = W * H
    diag_sizes = []
    for f in range(W + H - 1):
        v_min = max(0, f - W + 1)
        v_max = min(f, H - 1)
        diag_sizes.append(v_max - v_min + 1)
    cumulative = [0]
    for s in diag_sizes:
        cumulative.append(cumulative[-1] + s)
    b = [0,
         cumulative[1],
         cumulative[min(3, len(cumulative)-1)],
         cumulative[min(7, len(cumulative)-1)],
         N]
    return sorted(set(b))
```

### C.3 Reference Scan Tables

**4×4** (linear indices, v × 4 + u):

```
 0,  1,  4,  2,  5,  8,  3,  6,  9, 12,  7, 10, 13, 11, 14, 15
```

**8×8** (linear indices, v × 8 + u):

```
 0,  1,  8,  2,  9, 16,  3, 10, 17, 24,  4, 11, 18, 25, 32,  5,
12, 19, 26, 33, 40,  6, 13, 20, 27, 34, 41, 48,  7, 14, 21, 28,
35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51,
58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 55, 62, 63
```

Scan tables for larger blocks and non-square shapes are generated by the algorithm in §C.1.

---

## Appendix D: Default Filter Weights

### D.1 Constants

`FILTER_SHIFT` = 10. A weight of 1024 (= 2¹⁰) at the centre tap passes the input unchanged.

### D.2 Weight Layout

```
weights[layer][c_out][c_in][ky][kx]     // int12 signed
biases[layer][c_out]                     // int12 signed
```

Enumeration order for signalling (§8.5): layer 1 weights (36), layer 1 biases (4), layer 2 weights (144), layer 2 biases (4), layer 3 weights (144), layer 3 biases (4), layer 4 weights (36), layer 4 biases (1). Within each weight block: c_out varies slowest, then c_in, then ky, then kx varies fastest.

### D.3 Default Weights (Identity)

The default weights implement identity pass-through. All weights are 0 except the centre-tap self-connections:

```
Layer 1: w[0][0][1][1] = 1024              (all other w = 0, all b = 0)
Layer 2: w[c][c][1][1] = 1024 for c=0..3   (all other w = 0, all b = 0)
Layer 3: w[c][c][1][1] = 1024 for c=0..3   (all other w = 0, all b = 0)
Layer 4: w[0][0][1][1] = 1024              (all other w = 0, all b = 0)
```

These are placeholder values. Production default weights should be trained on coded video content and will replace these values in a future revision.

The same defaults are used for both luma and chroma planes.

---

## Appendix E: QP-to-Qstep Mapping

### E.1 Generating Formula

```python
def qstep(qp):
    base = [26, 29, 32, 36, 40, 45]
    return base[qp % 6] << (qp // 6)
```

### E.2 QP Table

| QP | Qstep | QP | Qstep | QP | Qstep | QP | Qstep |
|----|-------|----|-------|----|-------|----|-------|
| 0 | 26 | 13 | 116 | 26 | 512 | 39 | 2304 |
| 1 | 29 | 14 | 128 | 27 | 576 | 40 | 2560 |
| 2 | 32 | 15 | 144 | 28 | 640 | 41 | 2880 |
| 3 | 36 | 16 | 160 | 29 | 720 | 42 | 3328 |
| 4 | 40 | 17 | 180 | 30 | 832 | 43 | 3712 |
| 5 | 45 | 18 | 208 | 31 | 928 | 44 | 4096 |
| 6 | 52 | 19 | 232 | 32 | 1024 | 45 | 4608 |
| 7 | 58 | 20 | 256 | 33 | 1152 | 46 | 5120 |
| 8 | 64 | 21 | 288 | 34 | 1280 | 47 | 5760 |
| 9 | 72 | 22 | 320 | 35 | 1440 | 48 | 6656 |
| 10 | 80 | 23 | 360 | 36 | 1664 | 49 | 7424 |
| 11 | 90 | 24 | 416 | 37 | 1856 | 50 | 8192 |
| 12 | 104 | 25 | 464 | 38 | 2048 | 51 | 9216 |

QP range: 0 to 51. `block_qp` shall be clamped to [0, 51].

---

## Appendix F: Perceptual Frequency Weight Tables

### F.1 Generating Formula

```python
def perceptual_weight(i, j):
    """Perceptual frequency weight for coefficient position (i, j).
    i = row index (vertical frequency), j = column index (horizontal frequency).
    Both zero-based."""
    return min(16 + i * i + j * j, 112)
```

### F.2 Reference Tables

**4×4** (PW[i][j], i=row, j=column):

```
 16  17  20  25
 17  18  21  26
 20  21  24  29
 25  26  29  34
```

**8×8**:

```
 16  17  20  25  32  41  52  65
 17  18  21  26  33  42  53  66
 20  21  24  29  36  45  56  69
 25  26  29  34  41  50  61  74
 32  33  36  41  48  57  68  81
 41  42  45  50  57  66  77  90
 52  53  56  61  68  77  88 101
 65  66  69  74  81  90 101 112
```

**16×16 and 32×32:** Generated by the formula in §F.1. Values saturate at 112 for positions where i² + j² ≥ 96 (e.g. position (7, 7) and all higher-frequency positions in larger transforms).

### F.3 Properties

| Transform | DC weight | Corner weight | Max ratio |
|-----------|-----------|---------------|-----------|
| 4×4 | 16 | 34 | 2.1× |
| 8×8 | 16 | 112 | 7.0× |
| 16×16 | 16 | 112 | 7.0× |
| 32×32 | 16 | 112 | 7.0× |

The 8×8 corner ratio of 7.0 closely matches the JPEG Annex K luminance table's 7.6:1 ratio at position (7, 7), which was derived from psychovisual experiments on DCT basis function visibility (Peterson, Ahumada & Watson, 1993). The Lattice formula achieves comparable perceptual shaping from first principles without copying specific matrix values.
