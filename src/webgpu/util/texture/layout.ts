import * as C from '../../../common/constants.js';
import { assert, unreachable } from '../../../common/framework/util/util.js';
import { kTextureFormatInfo } from '../../capability_info.js';
import { align, isAligned } from '../math.js';

export const kBytesPerRowAlignment = 256;
export const kBufferCopyAlignment = 4;

export interface LayoutOptions {
  mipLevel: number;
  bytesPerRow?: number;
  rowsPerImage?: number;
}

const kDefaultLayoutOptions = { mipLevel: 0, bytesPerRow: undefined, rowsPerImage: undefined };

export function getMipSizePassthroughLayers(
  dimension: GPUTextureDimension,
  size: [number, number, number],
  mipLevel: number
): [number, number, number] {
  const shiftMinOne = (n: number) => Math.max(1, n >> mipLevel);
  switch (dimension) {
    case '1d':
      assert(size[2] === 1);
      return [shiftMinOne(size[0]), size[1], size[2]];
    case '2d':
      return [shiftMinOne(size[0]), shiftMinOne(size[1]), size[2]];
    case '3d':
      return [shiftMinOne(size[0]), shiftMinOne(size[1]), shiftMinOne(size[2])];
    default:
      unreachable();
  }
}

export function getTextureCopyLayout(
  format: GPUTextureFormat,
  dimension: GPUTextureDimension,
  size: [number, number, number],
  options: LayoutOptions = kDefaultLayoutOptions
): {
  bytesPerBlock: number;
  byteLength: number;
  minBytesPerRow: number;
  bytesPerRow: number;
  rowsPerImage: number;
} {
  const { mipLevel } = options;
  let { bytesPerRow, rowsPerImage } = options;

  const mipSize = getMipSizePassthroughLayers(dimension, size, mipLevel);

  const { blockWidth, blockHeight, bytesPerBlock } = kTextureFormatInfo[format];
  assert(!!bytesPerBlock && !!blockWidth && !!blockHeight);

  assert(isAligned(mipSize[0], blockWidth));
  const minBytesPerRow = (mipSize[0] / blockWidth) * bytesPerBlock;
  const alignedMinBytesPerRow = align(minBytesPerRow, kBytesPerRowAlignment);
  if (bytesPerRow !== undefined) {
    assert(bytesPerRow >= alignedMinBytesPerRow);
    assert(isAligned(bytesPerRow, kBytesPerRowAlignment));
  } else {
    bytesPerRow = alignedMinBytesPerRow;
  }

  if (rowsPerImage !== undefined) {
    assert(rowsPerImage >= mipSize[1]);
  } else {
    rowsPerImage = mipSize[1];
  }

  assert(isAligned(rowsPerImage, blockHeight));
  const bytesPerSlice = bytesPerRow * (rowsPerImage / blockHeight);
  const sliceSize =
    bytesPerRow * (mipSize[1] / blockHeight - 1) + bytesPerBlock * (mipSize[0] / blockWidth);
  const byteLength = bytesPerSlice * (mipSize[2] - 1) + sliceSize;

  return {
    bytesPerBlock,
    byteLength: align(byteLength, kBufferCopyAlignment),
    minBytesPerRow,
    bytesPerRow,
    rowsPerImage,
  };
}

export function fillTextureDataWithTexelValue(
  texelValue: ArrayBuffer,
  format: GPUTextureFormat,
  dimension: GPUTextureDimension,
  outputBuffer: ArrayBuffer,
  size: [number, number, number],
  options: LayoutOptions = kDefaultLayoutOptions
): void {
  const { blockWidth, blockHeight, bytesPerBlock } = kTextureFormatInfo[format];
  assert(!!bytesPerBlock && !!blockWidth && !!blockHeight);
  assert(bytesPerBlock === texelValue.byteLength);

  const { byteLength, rowsPerImage, bytesPerRow } = getTextureCopyLayout(
    format,
    dimension,
    size,
    options
  );

  assert(byteLength <= outputBuffer.byteLength);

  const mipSize = getMipSizePassthroughLayers(dimension, size, options.mipLevel);

  const texelValueBytes = new Uint8Array(texelValue);
  const outputTexelValueBytes = new Uint8Array(outputBuffer);
  for (let slice = 0; slice < mipSize[2]; ++slice) {
    for (let row = 0; row < mipSize[1]; row += blockHeight) {
      for (let col = 0; col < mipSize[0]; col += blockWidth) {
        const byteOffset =
          slice * rowsPerImage * bytesPerRow + row * bytesPerRow + col * texelValue.byteLength;
        outputTexelValueBytes.set(texelValueBytes, byteOffset);
      }
    }
  }
}

export function createTextureUploadBuffer(
  texelValue: ArrayBuffer,
  device: GPUDevice,
  format: GPUTextureFormat,
  dimension: GPUTextureDimension,
  size: [number, number, number],
  options: LayoutOptions = kDefaultLayoutOptions
): {
  buffer: GPUBuffer;
  bytesPerRow: number;
  rowsPerImage: number;
} {
  const { byteLength, bytesPerRow, rowsPerImage, bytesPerBlock } = getTextureCopyLayout(
    format,
    dimension,
    size,
    options
  );

  const [buffer, mapping] = device.createBufferMapped({
    size: byteLength,
    usage: C.BufferUsage.CopySrc,
  });

  assert(texelValue.byteLength === bytesPerBlock);
  fillTextureDataWithTexelValue(texelValue, format, dimension, mapping, size, options);
  buffer.unmap();

  return {
    buffer,
    bytesPerRow,
    rowsPerImage,
  };
}