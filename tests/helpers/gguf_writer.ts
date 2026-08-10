import fs from 'fs';

/**
 * Minimal GGUF v3 writer used by the test-suite.
 *
 * It emits a real file in the real binary layout, so the parser under test is
 * exercised against the same bytes llama.cpp would produce — not a stub.
 */

const GGUF_MAGIC = 0x46554747;
const ALIGNMENT = 32;

enum ValueType {
  UINT32 = 4,
  STRING = 8,
}

class ByteWriter {
  private chunks: Buffer[] = [];
  private length = 0;

  push(buf: Buffer): void {
    this.chunks.push(buf);
    this.length += buf.length;
  }

  u32(value: number): void {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(value, 0);
    this.push(b);
  }

  u64(value: number | bigint): void {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(value), 0);
    this.push(b);
  }

  str(value: string): void {
    const bytes = Buffer.from(value, 'utf8');
    this.u64(bytes.length);
    this.push(bytes);
  }

  get size(): number {
    return this.length;
  }

  toBuffer(): Buffer {
    return Buffer.concat(this.chunks);
  }
}

export interface SyntheticTensor {
  name: string;
  /** Q8_0-style layout is avoided; F32 keeps size math trivial and exact. */
  elementCount: number;
}

export interface WrittenGguf {
  filePath: string;
  blockCount: number;
  tensors: SyntheticTensor[];
  tensorDataOffset: number;
  totalTensorBytes: number;
}

/**
 * Write a small but structurally valid GGUF containing `blockCount` layers,
 * each with one F32 tensor filled with a deterministic byte pattern so reads
 * can be verified downstream.
 */
export function writeSyntheticGguf(
  filePath: string,
  options: { blockCount?: number; elementsPerTensor?: number; architecture?: string } = {}
): WrittenGguf {
  const blockCount = options.blockCount ?? 4;
  const elementsPerTensor = options.elementsPerTensor ?? 256;
  const architecture = options.architecture ?? 'llama';

  const tensors: SyntheticTensor[] = [];
  for (let layer = 0; layer < blockCount; layer++) {
    tensors.push({ name: `blk.${layer}.attn_q.weight`, elementCount: elementsPerTensor });
  }

  const metadata: Array<[string, ValueType, string | number]> = [
    ['general.architecture', ValueType.STRING, architecture],
    ['general.name', ValueType.STRING, 'synthetic-test-model'],
    [`${architecture}.block_count`, ValueType.UINT32, blockCount],
    [`${architecture}.context_length`, ValueType.UINT32, 2048],
    [`${architecture}.embedding_length`, ValueType.UINT32, 64],
    [`${architecture}.attention.head_count`, ValueType.UINT32, 4],
    [`${architecture}.attention.head_count_kv`, ValueType.UINT32, 2],
    ['general.alignment', ValueType.UINT32, ALIGNMENT],
  ];

  const header = new ByteWriter();
  header.u32(GGUF_MAGIC);
  header.u32(3);
  header.u64(tensors.length);
  header.u64(metadata.length);

  for (const [key, type, value] of metadata) {
    header.str(key);
    header.u32(type);
    if (type === ValueType.STRING) header.str(String(value));
    else header.u32(Number(value));
  }

  // Tensor info records; F32 (ggml type 0) => 4 bytes per element.
  let runningOffset = 0;
  const tensorOffsets: number[] = [];
  for (const tensor of tensors) {
    header.str(tensor.name);
    header.u32(1); // n_dims
    header.u64(tensor.elementCount);
    header.u32(0); // ggml_type F32
    header.u64(runningOffset);
    tensorOffsets.push(runningOffset);
    runningOffset += tensor.elementCount * 4;
  }

  const headerBuf = header.toBuffer();
  const tensorDataOffset = Math.ceil(headerBuf.length / ALIGNMENT) * ALIGNMENT;
  const padding = Buffer.alloc(tensorDataOffset - headerBuf.length, 0);

  // Deterministic payload: layer N is filled with byte value N+1.
  const payloads = tensors.map((tensor, index) => Buffer.alloc(tensor.elementCount * 4, index + 1));

  fs.writeFileSync(filePath, Buffer.concat([headerBuf, padding, ...payloads]));

  return {
    filePath,
    blockCount,
    tensors,
    tensorDataOffset,
    totalTensorBytes: runningOffset,
  };
}
