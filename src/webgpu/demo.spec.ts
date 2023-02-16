export const description = `Test compute shader builtin variables`;

import { makeTestGroup } from '../common/framework/test_group.js';
import { iterRange } from '../common/util/util.js';
import { GPUTest } from './gpu_test.js';

export const g = makeTestGroup(GPUTest);

g.test('demo')
  .params(u => u.beginSubcases())
  .fn(async t => {
    const num_elements = 1024;
    const wgsize = 32;
    const groups = (num_elements + wgsize - 1) / wgsize;

    const wgsl = `
      @group(0) @binding(0)
      var<storage, read_write> input : array<u32>;

      @group(0) @binding(1)
      var<storage, read_write> output : array<u32>;

      const wgsize = ${wgsize};

      var<workgroup> scratch : array<u32, wgsize>;

      @compute @workgroup_size(wgsize)
      fn main(@builtin(local_invocation_index) idx   : u32,
              @builtin(workgroup_id)           group : vec3<u32>) {
        var sum = 0u;
        for (var base = 0u; base < ${num_elements}; base+=wgsize) {
          workgroupBarrier();
          scratch[idx] = input[base + idx];
          workgroupBarrier();
          for (var i = 0u; i < wgsize; i++) {
            sum += scratch[i];
          }
        }
        let global_idx = group.x * wgsize + idx;
        output[global_idx] = sum - input[global_idx];
      }
    `;

    const pipeline = t.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: t.device.createShaderModule({
          code: wgsl,
        }),
        entryPoint: 'main',
      },
    });

    // Set up the buffer and bind group.
    const inputBuffer = t.makeBufferWithContents(
      new Uint32Array([...iterRange(num_elements, x => x)]),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const outputBuffer = t.makeBufferWithContents(
      new Uint32Array([...iterRange(num_elements, _ => 0)]),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const bindGroup = t.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
      ],
    });

    // Run the shader.
    const encoder = t.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups);
    pass.end();
    t.queue.submit([encoder.finish()]);

    t.expectGPUBufferValuesEqual(
      outputBuffer,
      new Uint32Array([...iterRange(num_elements, x => (num_elements * (num_elements - 1)) / 2 - x)])
    );
  });
