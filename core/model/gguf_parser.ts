import fs from 'fs';

// ============================================================================
// GGUF Binary Format Constants
// ============================================================================

const GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian

/** ggml_type enum — maps integer type ID to human-readable name and byte-per-element info */
export const GGML_TYPES: Record<number, { name: string; blockSize: number; bytesPerBlock: number }> = {
  0:  { name: 'F32',     blockSize: 1,   bytesPerBlock: 4 },
  1:  { name: 'F16',     blockSize: 1,   bytesPerBlock: 2 },
  2:  { name: 'Q4_0',    blockSize: 32,  bytesPerBlock: 18 },    // 32 values → 2 byte scale + 16 byte quants
  3:  { name: 'Q4_1',    blockSize: 32,  bytesPerBlock: 20 },
  6:  { name: 'Q5_0',    blockSize: 32,  bytesPerBlock: 22 },
  7:  { name: 'Q5_1',    blockSize: 32,  bytesPerBlock: 24 },
  8:  { name: 'Q8_0',    blockSize: 32,  bytesPerBlock: 34 },
  9:  { name: 'Q8_1',    blockSize: 32,  bytesPerBlock: 36 },
  10: { name: 'Q2_K',    blockSize: 256, bytesPerBlock: 84 },
  11: { name: 'Q3_K',    blockSize: 256, bytesPerBlock: 110 },
  12: { name: 'Q4_K',    blockSize: 256, bytesPerBlock: 144 },
  13: { name: 'Q5_K',    blockSize: 256, bytesPerBlock: 176 },
  14: { name: 'Q6_K',    blockSize: 256, bytesPerBlock: 210 },
  15: { name: 'Q8_K',    blockSize: 256, bytesPerBlock: 292 },
  16: { name: 'IQ2_XXS', blockSize: 256, bytesPerBlock: 66 },
  17: { name: 'IQ2_XS',  blockSize: 256, bytesPerBlock: 74 },
  18: { name: 'IQ3_XXS', blockSize: 256, bytesPerBlock: 98 },
  19: { name: 'IQ1_S',   blockSize: 256, bytesPerBlock: 50 },
  20: { name: 'IQ4_NL',  blockSize: 32,  bytesPerBlock: 18 },
  21: { name: 'IQ3_S',   blockSize: 256, bytesPerBlock: 110 },
  22: { name: 'IQ2_S',   blockSize: 256, bytesPerBlock: 82 },
  23: { name: 'IQ4_XS',  blockSize: 256, bytesPerBlock: 136 },
  24: { name: 'I8',      blockSize: 1,   bytesPerBlock: 1 },
  25: { name: 'I16',     blockSize: 1,   bytesPerBlock: 2 },
  26: { name: 'I32',     blockSize: 1,   bytesPerBlock: 4 },
  27: { name: 'I64',     blockSize: 1,   bytesPerBlock: 8 },
  28: { name: 'F64',     blockSize: 1,   bytesPerBlock: 8 },
  29: { name: 'IQ1_M',   blockSize: 256, bytesPerBlock: 56 },
  30: { name: 'BF16',    blockSize: 1,   bytesPerBlock: 2 },
};

/** GGUF metadata value types */
enum GGUFValueType {
  UINT8   = 0,
  INT8    = 1,
  UINT16  = 2,
  INT16   = 3,
  UINT32  = 4,
  INT32   = 5,
  FLOAT32 = 6,
  BOOL    = 7,
  STRING  = 8,
  ARRAY   = 9,
  UINT64  = 10,
  INT64   = 11,
  FLOAT64 = 12,
}

// ============================================================================
// GGUF Parsed Data Types
// ============================================================================

export interface GgufTensorInfo {
  name: string;
  ndims: number;
  dims: bigint[];
  ggmlType: number;
  ggmlTypeName: string;
  offset: bigint;        // Offset from start of tensor data section
  absoluteOffset: number; // Absolute offset in the file
  sizeBytes: number;     // Calculated size of this tensor in bytes
  numElements: bigint;   // Total number of elements
  layerIndex: number;    // Extracted layer index (-1 if not a layer tensor)
}

export interface GgufMetadata {
  filename: string;
  filePath: string;
  fileSizeBytes: number;

  // Header fields
  version: number;
  tensorCount: number;
  metadataKvCount: number;

  // Extracted from metadata KV pairs
  architecture: string;
  modelName: string;
  contextLength: number;
  embeddingLength: number;
  blockCount: number;
  headCount: number;
  headCountKv: number;
  quantization: string;               // Dominant quantization type name
  parameterCountBillions: number;

  // Calculated
  tensorDataOffset: number;            // Where tensor binary data starts in the file
  totalTensorDataBytes: number;        // Total bytes of all tensor data
  estimatedRamRequiredBytes: number;
  estimatedVramRequiredBytes: number;
  estimatedStorageRequiredBytes: number;

  // Full metadata map and tensor list
  metadataMap: Map<string, any>;
  tensors: GgufTensorInfo[];
}

// ============================================================================
// Binary Reader Helper
// ============================================================================

class BinaryReader {
  private fd: number;
  private pos: number = 0;
  private buf: Buffer;
  private bufferSize: number;

  constructor(fd: number, initialBufferSize = 64 * 1024) {
    this.fd = fd;
    this.bufferSize = initialBufferSize;
    this.buf = Buffer.alloc(initialBufferSize);
  }

  get position(): number { return this.pos; }
  set position(p: number) { this.pos = p; }

  private readBytes(count: number): Buffer {
    if (count > this.bufferSize) {
      this.buf = Buffer.alloc(count);
      this.bufferSize = count;
    }
    const bytesRead = fs.readSync(this.fd, this.buf, 0, count, this.pos);
    if (bytesRead < count) {
      throw new Error(`Unexpected EOF: wanted ${count} bytes at offset ${this.pos}, got ${bytesRead}`);
    }
    this.pos += count;
    return this.buf;
  }

  readUint8(): number {
    const b = this.readBytes(1);
    return b.readUInt8(0);
  }

  readInt8(): number {
    const b = this.readBytes(1);
    return b.readInt8(0);
  }

  readUint16(): number {
    const b = this.readBytes(2);
    return b.readUInt16LE(0);
  }

  readInt16(): number {
    const b = this.readBytes(2);
    return b.readInt16LE(0);
  }

  readUint32(): number {
    const b = this.readBytes(4);
    return b.readUInt32LE(0);
  }

  readInt32(): number {
    const b = this.readBytes(4);
    return b.readInt32LE(0);
  }

  readUint64(): bigint {
    const b = this.readBytes(8);
    return b.readBigUInt64LE(0);
  }

  readInt64(): bigint {
    const b = this.readBytes(8);
    return b.readBigInt64LE(0);
  }

  readFloat32(): number {
    const b = this.readBytes(4);
    return b.readFloatLE(0);
  }

  readFloat64(): number {
    const b = this.readBytes(8);
    return b.readDoubleLE(0);
  }

  readBool(): boolean {
    return this.readUint8() !== 0;
  }

  readString(): string {
    const length = this.readUint64();
    const len = Number(length);
    if (len === 0) return '';
    if (len > 1024 * 1024) {
      throw new Error(`String too long: ${len} bytes at offset ${this.pos}`);
    }
    const strBuf = Buffer.alloc(len);
    const bytesRead = fs.readSync(this.fd, strBuf, 0, len, this.pos);
    this.pos += len;
    return strBuf.toString('utf8', 0, bytesRead);
  }

  readMetadataValue(valueType: GGUFValueType): any {
    switch (valueType) {
      case GGUFValueType.UINT8:   return this.readUint8();
      case GGUFValueType.INT8:    return this.readInt8();
      case GGUFValueType.UINT16:  return this.readUint16();
      case GGUFValueType.INT16:   return this.readInt16();
      case GGUFValueType.UINT32:  return this.readUint32();
      case GGUFValueType.INT32:   return this.readInt32();
      case GGUFValueType.FLOAT32: return this.readFloat32();
      case GGUFValueType.BOOL:    return this.readBool();
      case GGUFValueType.STRING:  return this.readString();
      case GGUFValueType.UINT64:  return Number(this.readUint64());
      case GGUFValueType.INT64:   return Number(this.readInt64());
      case GGUFValueType.FLOAT64: return this.readFloat64();
      case GGUFValueType.ARRAY: {
        const elemType = this.readUint32() as GGUFValueType;
        const count = Number(this.readUint64());
        const arr: any[] = [];
        // For very large arrays (e.g. tokenizer), limit reading to first 100 elements
        const maxRead = Math.min(count, 100);
        for (let i = 0; i < maxRead; i++) {
          arr.push(this.readMetadataValue(elemType));
        }
        // Skip remaining elements
        if (count > maxRead) {
          for (let i = maxRead; i < count; i++) {
            this.skipMetadataValue(elemType);
          }
        }
        return arr;
      }
      default:
        throw new Error(`Unknown GGUF value type: ${valueType} at offset ${this.pos}`);
    }
  }

  /** Skip a metadata value without allocating memory for large arrays */
  private skipMetadataValue(valueType: GGUFValueType): void {
    switch (valueType) {
      case GGUFValueType.UINT8:
      case GGUFValueType.INT8:
      case GGUFValueType.BOOL:
        this.pos += 1; break;
      case GGUFValueType.UINT16:
      case GGUFValueType.INT16:
        this.pos += 2; break;
      case GGUFValueType.UINT32:
      case GGUFValueType.INT32:
      case GGUFValueType.FLOAT32:
        this.pos += 4; break;
      case GGUFValueType.UINT64:
      case GGUFValueType.INT64:
      case GGUFValueType.FLOAT64:
        this.pos += 8; break;
      case GGUFValueType.STRING: {
        const len = Number(this.readUint64());
        this.pos += len;
        break;
      }
      case GGUFValueType.ARRAY: {
        const elemType = this.readUint32() as GGUFValueType;
        const count = Number(this.readUint64());
        for (let i = 0; i < count; i++) {
          this.skipMetadataValue(elemType);
        }
        break;
      }
      default:
        throw new Error(`Cannot skip unknown GGUF value type: ${valueType}`);
    }
  }
}

// ============================================================================
// Calculate Tensor Size in Bytes
// ============================================================================

function calculateTensorSizeBytes(ggmlType: number, dims: bigint[]): number {
  const typeInfo = GGML_TYPES[ggmlType];
  if (!typeInfo) {
    // Unknown type — estimate as F16
    const numElements = dims.reduce((a, b) => a * b, 1n);
    return Number(numElements) * 2;
  }

  const numElements = dims.reduce((a, b) => a * b, 1n);
  const ne = Number(numElements);

  if (typeInfo.blockSize === 1) {
    return ne * typeInfo.bytesPerBlock;
  }

  // Quantized: number of blocks × bytes per block
  const numBlocks = Math.ceil(ne / typeInfo.blockSize);
  return numBlocks * typeInfo.bytesPerBlock;
}

/** Extract layer index from tensor name like "blk.42.attn_q.weight" → 42 */
function extractLayerIndex(tensorName: string): number {
  const match = tensorName.match(/blk\.(\d+)\./);
  if (match) return parseInt(match[1], 10);
  return -1; // Non-layer tensor (e.g., "output.weight", "token_embd.weight")
}

// ============================================================================
// Main GGUF Parser — Reads Real Binary Format
// ============================================================================

export function parseGgufHeader(filePath: string): GgufMetadata {
  const filename = filePath.split(/[/\\]/).pop() || 'model.gguf';

  let stats: fs.Stats;
  try {
    stats = fs.statSync(filePath);
  } catch {
    // Never invent metadata for a file we cannot read — the caller must know.
    throw new Error(`Model file not found or unreadable: ${filePath}`);
  }

  const fileSizeBytes = stats.size;
  const fd = fs.openSync(filePath, 'r');
  const reader = new BinaryReader(fd);

  try {
    // ---- 1. Read Header ----
    const magic = reader.readUint32();
    if (magic !== GGUF_MAGIC) {
      throw new Error(
        `Invalid GGUF magic number: 0x${magic.toString(16).toUpperCase()}. ` +
        `Expected 0x${GGUF_MAGIC.toString(16).toUpperCase()} ("GGUF"). ` +
        `This file is not a valid GGUF model.`
      );
    }

    const version = reader.readUint32();
    if (version < 2 || version > 3) {
      console.warn(`GGUF version ${version} — only v2 and v3 are well-tested. Proceeding anyway.`);
    }

    const tensorCount = Number(reader.readUint64());
    const metadataKvCount = Number(reader.readUint64());

    // ---- 2. Read Metadata Key-Value Pairs ----
    const metadataMap = new Map<string, any>();

    for (let i = 0; i < metadataKvCount; i++) {
      const key = reader.readString();
      const valueType = reader.readUint32() as GGUFValueType;
      const value = reader.readMetadataValue(valueType);
      metadataMap.set(key, value);
    }

    // ---- 3. Extract Model Metadata ----
    const architecture = (metadataMap.get('general.architecture') as string) || 'unknown';
    const modelName = (metadataMap.get('general.name') as string) || filename;
    const alignment = (metadataMap.get('general.alignment') as number) || 32;

    // Architecture-specific metadata keys
    const arch = architecture;
    const contextLength = (metadataMap.get(`${arch}.context_length`) as number) || 4096;
    const embeddingLength = (metadataMap.get(`${arch}.embedding_length`) as number) || 4096;
    const blockCount = (metadataMap.get(`${arch}.block_count`) as number) || 32;
    const headCount = (metadataMap.get(`${arch}.attention.head_count`) as number) || 32;
    const headCountKv = (metadataMap.get(`${arch}.attention.head_count_kv`) as number) || headCount;

    // ---- 4. Read Tensor Info Array ----
    const tensors: GgufTensorInfo[] = [];
    const quantTypeCounts = new Map<number, number>();

    for (let i = 0; i < tensorCount; i++) {
      const name = reader.readString();
      const ndims = reader.readUint32();
      const dims: bigint[] = [];
      for (let d = 0; d < ndims; d++) {
        dims.push(reader.readUint64());
      }
      const ggmlType = reader.readUint32();
      const offset = reader.readUint64();

      const typeInfo = GGML_TYPES[ggmlType];
      const ggmlTypeName = typeInfo ? typeInfo.name : `UNKNOWN_${ggmlType}`;
      const numElements = dims.reduce((a, b) => a * b, 1n);
      const sizeBytes = calculateTensorSizeBytes(ggmlType, dims);
      const layerIndex = extractLayerIndex(name);

      tensors.push({
        name,
        ndims,
        dims,
        ggmlType,
        ggmlTypeName,
        offset,
        absoluteOffset: 0, // Will be calculated after we know tensor data start
        sizeBytes,
        numElements,
        layerIndex,
      });

      // Track how many *bytes* each type accounts for. Counting tensors instead
      // would label a Q4_K model as F32, because the dozens of tiny F32 norm
      // tensors outnumber the handful of huge quantised weight matrices.
      quantTypeCounts.set(ggmlType, (quantTypeCounts.get(ggmlType) || 0) + sizeBytes);
    }

    // ---- 5. Calculate Tensor Data Offset ----
    // After the tensor info array, data is aligned to `alignment` boundary
    const tensorInfoEnd = reader.position;
    const tensorDataOffset = Math.ceil(tensorInfoEnd / alignment) * alignment;

    // Set absolute offsets for each tensor
    for (const t of tensors) {
      t.absoluteOffset = tensorDataOffset + Number(t.offset);
    }

    // ---- 6. Determine Dominant Quantization (by bytes, see above) ----
    let dominantType = 0;
    let maxBytes = -1;
    for (const [typeId, bytes] of quantTypeCounts.entries()) {
      if (bytes > maxBytes) {
        maxBytes = bytes;
        dominantType = typeId;
      }
    }
    const quantization = GGML_TYPES[dominantType]?.name || 'UNKNOWN';

    // ---- 7. Calculate Model Statistics ----
    const totalTensorDataBytes = tensors.reduce((sum, t) => sum + t.sizeBytes, 0);

    // Estimate parameter count from total elements of weight tensors
    const totalElements = tensors.reduce((sum, t) => sum + Number(t.numElements), 0);
    const parameterCountBillions = Number((totalElements / 1e9).toFixed(2));

    // Memory requirements
    const estimatedStorageRequiredBytes = fileSizeBytes;
    const estimatedRamRequiredBytes = Math.ceil(totalTensorDataBytes * 1.15); // 15% overhead for workspace
    const estimatedVramRequiredBytes = Math.ceil(totalTensorDataBytes * 0.25); // Minimum working window

    return {
      filename,
      filePath,
      fileSizeBytes,
      version,
      tensorCount,
      metadataKvCount,
      architecture,
      modelName,
      contextLength,
      embeddingLength,
      blockCount,
      headCount,
      headCountKv,
      quantization,
      parameterCountBillions,
      tensorDataOffset,
      totalTensorDataBytes,
      estimatedRamRequiredBytes,
      estimatedVramRequiredBytes,
      estimatedStorageRequiredBytes,
      metadataMap,
      tensors,
    };
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Build GGUF metadata for a synthetic in-memory model. Used only by tests that
 * need a tensor layout without a multi-gigabyte file on disk; it is never a
 * fallback for a real parse failure.
 */
export function createSyntheticGgufMetadata(
  filePath: string,
  options: { blockCount?: number; tensorBytes?: number } = {}
): GgufMetadata {
  const filename = filePath.split(/[/\\]/).pop() || 'synthetic.gguf';
  const blockCount = options.blockCount ?? 8;
  const tensorBytes = options.tensorBytes ?? 4096;
  const tensors: GgufTensorInfo[] = [];

  for (let l = 0; l < blockCount; l++) {
    tensors.push({
      name: `blk.${l}.attn_q.weight`,
      ndims: 2,
      dims: [64n, 64n],
      ggmlType: 12,
      ggmlTypeName: 'Q4_K',
      offset: BigInt(l * tensorBytes),
      absoluteOffset: 0,
      sizeBytes: tensorBytes,
      numElements: 64n * 64n,
      layerIndex: l,
    });
  }

  const totalTensorDataBytes = tensors.reduce((sum, t) => sum + t.sizeBytes, 0);

  return {
    filename,
    filePath,
    fileSizeBytes: totalTensorDataBytes,
    version: 3,
    tensorCount: tensors.length,
    metadataKvCount: 0,
    architecture: 'synthetic',
    modelName: filename,
    contextLength: 2048,
    embeddingLength: 64,
    blockCount,
    headCount: 4,
    headCountKv: 4,
    quantization: 'Q4_K',
    parameterCountBillions: 0,
    tensorDataOffset: 0,
    totalTensorDataBytes,
    estimatedRamRequiredBytes: Math.ceil(totalTensorDataBytes * 1.15),
    estimatedVramRequiredBytes: Math.ceil(totalTensorDataBytes * 0.25),
    estimatedStorageRequiredBytes: totalTensorDataBytes,
    metadataMap: new Map(),
    tensors,
  };
}
