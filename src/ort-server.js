/**
 * Lightweight ONNX Runtime wrapper for PartyKit/Cloudflare Workers.
 *
 * Bypasses the ort npm package entirely, directly calling the Emscripten
 * module's C API. Uses the same `instantiateWasm` pattern as PhysX.
 *
 * Key constraints addressed:
 *  - numThreads=1 to prevent Worker spawning
 *  - Static WASM import via esbuild loader
 *  - No dynamic import() calls
 */

// Polyfill performance for Cloudflare Workers
// Emscripten uses performance.timeOrigin + performance.now() and converts to BigInt
if (typeof performance === 'undefined') {
  const _t0 = Date.now();
  globalThis.performance = { now: () => Date.now() - _t0, timeOrigin: _t0 };
} else {
  if (typeof performance.now !== 'function') {
    const _t0 = Date.now();
    performance.now = () => Date.now() - _t0;
  }
  if (typeof performance.timeOrigin !== 'number' || isNaN(performance.timeOrigin)) {
    performance.timeOrigin = Date.now();
  }
}

import ortFactory from '../assets/onnx/ort-wasm-simd-threaded.mjs';
import ortWasm from '../assets/onnx/ort-wasm-simd-threaded.wasm';

// ORT data type enum values (from onnxruntime C API)
const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1;

// Data location enum (0=none, 1=cpu, 2=cpu-pinned, 3=texture, 4=gpu-buffer, 5=ml-tensor)
const DATA_LOCATION_CPU = 1;

let _module = null;

/**
 * Initialize the ONNX Runtime WASM module.
 * Must be called once before any other operations.
 * @returns {Promise<object>} The Emscripten module instance
 */
export async function initOrt() {
  if (_module) return _module;

  // ortWasm is a pre-compiled WebAssembly.Module from static import
  _module = await ortFactory({
    numThreads: 1,
    instantiateWasm: (imports, callback) => {
      const instance = new WebAssembly.Instance(ortWasm, imports);
      callback(instance);
      return instance.exports;
    },
  });

  // Initialize ORT: numThreads=1, loggingLevel=3 (WARNING)
  const rc = _module._OrtInit(1, 3);
  if (rc !== 0) {
    throw new Error(`ORT init failed with code ${rc}`);
  }

  console.log('ONNX Runtime initialized (single-thread, WASM SIMD)');
  return _module;
}

/**
 * Helper: allocate a C string on the WASM heap.
 */
function allocString(mod, str) {
  const bytes = new TextEncoder().encode(str + '\0');
  const ptr = mod._malloc(bytes.length);
  mod.HEAPU8.set(bytes, ptr);
  return ptr;
}

/**
 * Create an inference session from a model buffer.
 * @param {ArrayBuffer|Uint8Array} modelData - The ONNX model bytes
 * @returns {object} Session object with handle and metadata
 */
export function createSession(modelData) {
  const mod = _module;
  if (!mod) throw new Error('ORT not initialized');

  const data = modelData instanceof Uint8Array ? modelData : new Uint8Array(modelData);

  // Copy model data to WASM heap
  const dataPtr = mod._malloc(data.byteLength);
  if (dataPtr === 0) throw new Error('Failed to allocate model buffer');
  mod.HEAPU8.set(data, dataPtr);

  // Create session options: graphOptLevel=3, enableCpuMemArena=true,
  // enableMemPattern=true, executionMode=0(sequential), enableProfiling=false,
  // profileFilePrefix=0, logId=0, logSeverityLevel=3, logVerbosityLevel=0,
  // optimizedModelFilePath=0
  const optionsHandle = mod._OrtCreateSessionOptions(3, true, true, 0, false, 0, 0, 3, 0, 0);
  if (optionsHandle === 0) throw new Error('Failed to create session options');

  // Create session
  let sessionHandle;
  try {
    sessionHandle = mod._OrtCreateSession(dataPtr, data.byteLength, optionsHandle);
  } finally {
    mod._OrtReleaseSessionOptions(optionsHandle);
    mod._free(dataPtr);
  }

  if (sessionHandle === 0) throw new Error('Failed to create session');

  // Get input/output count
  const stack = mod.stackSave();
  try {
    const inputCountPtr = mod.stackAlloc(4);
    const outputCountPtr = mod.stackAlloc(4);
    mod._OrtGetInputOutputCount(sessionHandle, inputCountPtr, outputCountPtr);
    const inputCount = mod.getValue(inputCountPtr, 'i32');
    const outputCount = mod.getValue(outputCountPtr, 'i32');

    // Get input/output names
    const inputNames = [];
    const inputNamesEncoded = [];
    for (let i = 0; i < inputCount; i++) {
      const namePtr = mod.stackAlloc(4);
      const metaPtr = mod.stackAlloc(4);
      mod._OrtGetInputOutputMetadata(sessionHandle, i, namePtr, metaPtr);
      const nameOffset = mod.getValue(namePtr, 'i32');
      const name = mod.UTF8ToString(nameOffset);
      inputNames.push(name);
      inputNamesEncoded.push(nameOffset);
    }

    const outputNames = [];
    const outputNamesEncoded = [];
    for (let i = 0; i < outputCount; i++) {
      const namePtr = mod.stackAlloc(4);
      const metaPtr = mod.stackAlloc(4);
      mod._OrtGetInputOutputMetadata(sessionHandle, inputCount + i, namePtr, metaPtr);
      const nameOffset = mod.getValue(namePtr, 'i32');
      const name = mod.UTF8ToString(nameOffset);
      outputNames.push(name);
      outputNamesEncoded.push(nameOffset);
    }

    console.log(`Session created: inputs=[${inputNames}], outputs=[${outputNames}]`);

    return {
      handle: sessionHandle,
      inputCount,
      outputCount,
      inputNames,
      outputNames,
      inputNamesEncoded,
      outputNamesEncoded,
    };
  } finally {
    mod.stackRestore(stack);
  }
}

/**
 * Run inference with float32 inputs and outputs.
 * @param {object} session - Session from createSession()
 * @param {Record<string, { data: Float32Array, dims: number[] }>} feeds - Input tensors
 * @returns {Record<string, Float32Array>} Output tensor data
 */
export function runSession(session, feeds) {
  const mod = _module;
  if (!mod) throw new Error('ORT not initialized');

  const ptrSize = 4; // 32-bit pointers
  const inputCount = session.inputCount;
  const outputCount = session.outputCount;

  const inputTensorHandles = [];
  const allocs = [];
  const beforeStack = mod.stackSave();

  try {
    // Create run options: logSeverity=3(WARNING), logVerbosity=0, terminate=false, tag=0
    const runOptionsHandle = mod._OrtCreateRunOptions(3, 0, false, 0);
    if (runOptionsHandle === 0) throw new Error('Failed to create run options');
    allocs.push(() => mod._OrtReleaseRunOptions(runOptionsHandle));

    // Allocate stack space for input/output pointers
    const inputValuesOffset = mod.stackAlloc(inputCount * ptrSize);
    const inputNamesOffset = mod.stackAlloc(inputCount * ptrSize);
    const outputValuesOffset = mod.stackAlloc(outputCount * ptrSize);
    const outputNamesOffset = mod.stackAlloc(outputCount * ptrSize);

    // Create input tensors
    for (let i = 0; i < inputCount; i++) {
      const name = session.inputNames[i];
      const feed = feeds[name];
      if (!feed) throw new Error(`Missing input feed: ${name}`);

      const { data, dims } = feed;
      const byteLength = data.byteLength;

      // Copy data to WASM heap
      const dataPtr = mod._malloc(byteLength);
      allocs.push(() => mod._free(dataPtr));
      mod.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, byteLength), dataPtr);

      // Create dims array on stack
      const dimsOffset = mod.stackAlloc(dims.length * ptrSize);
      for (let d = 0; d < dims.length; d++) {
        mod.setValue(dimsOffset + d * ptrSize, dims[d], 'i32');
      }

      const tensorHandle = mod._OrtCreateTensor(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        dataPtr,
        byteLength,
        dimsOffset,
        dims.length,
        DATA_LOCATION_CPU,
      );
      if (tensorHandle === 0) {
        // Check ORT error
        const errStack = mod.stackSave();
        const errCodePtr = mod.stackAlloc(4);
        const errMsgPtr = mod.stackAlloc(4);
        mod._OrtGetLastError(errCodePtr, errMsgPtr);
        const errCode = mod.getValue(errCodePtr, 'i32');
        const errMsgOffset = mod.getValue(errMsgPtr, '*');
        const errMsg = errMsgOffset ? mod.UTF8ToString(errMsgOffset) : 'unknown';
        mod.stackRestore(errStack);
        throw new Error(`Failed to create input tensor '${name}': ORT error ${errCode}: ${errMsg}`);
      }
      inputTensorHandles.push(tensorHandle);
      allocs.push(() => mod._OrtReleaseTensor(tensorHandle));

      mod.setValue(inputValuesOffset + i * ptrSize, tensorHandle, '*');
      mod.setValue(inputNamesOffset + i * ptrSize, session.inputNamesEncoded[i], '*');
    }

    // Set output names (with null tensor handles for auto-allocation)
    const outputTensorHandles = new Array(outputCount).fill(0);
    for (let i = 0; i < outputCount; i++) {
      mod.setValue(outputValuesOffset + i * ptrSize, 0, '*');
      mod.setValue(outputNamesOffset + i * ptrSize, session.outputNamesEncoded[i], '*');
    }

    // Run inference (synchronous — WASM export, not asyncify)
    const errorCode = mod._OrtRun(
      session.handle,
      inputNamesOffset,
      inputValuesOffset,
      inputCount,
      outputNamesOffset,
      outputCount,
      outputValuesOffset,
      runOptionsHandle,
    );

    if (errorCode !== 0) {
      throw new Error(`OrtRun failed with code ${errorCode}`);
    }

    // Read output tensors
    const results = {};
    for (let i = 0; i < outputCount; i++) {
      const outputTensorHandle = mod.getValue(outputValuesOffset + i * ptrSize, '*');
      if (outputTensorHandle === 0) throw new Error(`Output tensor ${i} is null`);

      const oStack = mod.stackSave();
      try {
        const dataTypePtr = mod.stackAlloc(4);
        const dataOffsetPtr = mod.stackAlloc(4);
        const dimsOffsetPtr = mod.stackAlloc(4);
        const dimsLengthPtr = mod.stackAlloc(4);

        // OrtGetTensorData: get the tensor's data pointer, type, dims
        mod.setValue(dataTypePtr, 0, 'i32');
        const rc = mod._OrtGetTensorData(
          outputTensorHandle,
          dataTypePtr,
          dataOffsetPtr,
          dimsOffsetPtr,
          dimsLengthPtr,
        );
        if (rc !== 0) throw new Error(`OrtGetTensorData failed: ${rc}`);

        const dataOffset = mod.getValue(dataOffsetPtr, '*');
        const dimsOffset = mod.getValue(dimsOffsetPtr, '*');
        const dimsLength = mod.getValue(dimsLengthPtr, 'i32');

        // Read dimensions
        const dims = [];
        let totalSize = 1;
        for (let d = 0; d < dimsLength; d++) {
          const dim = mod.getValue(dimsOffset + d * ptrSize, 'i32');
          dims.push(dim);
          totalSize *= dim;
        }

        // Copy float32 data out
        const output = new Float32Array(totalSize);
        const src = new Float32Array(mod.HEAPU8.buffer, dataOffset, totalSize);
        output.set(src);

        results[session.outputNames[i]] = output;
      } finally {
        mod.stackRestore(oStack);
      }

      mod._OrtReleaseTensor(outputTensorHandle);
    }

    return results;
  } finally {
    // Release allocated resources
    for (const cleanup of allocs) cleanup();
    mod.stackRestore(beforeStack);
  }
}
