/**
 * CTC Loss MPS Extension - Objective-C++ PyTorch Bindings
 *
 * This file provides the bridge between Python/PyTorch and Metal compute
 * shaders.
 *
 * =============================================================================
 * ARCHITECTURE OVERVIEW
 * =============================================================================
 *
 * Python (CTCLossMPS)
 *     |
 *     v
 * PyTorch Extension (this file)
 *     |
 *     v
 * Metal Performance Shaders (ctc_kernels.metal)
 *     |
 *     v
 * Apple Silicon GPU
 *
 * =============================================================================
 * KEY CONCEPTS
 * =============================================================================
 *
 * 1. MTLBuffer: Metal's GPU memory buffer. PyTorch MPS tensors have underlying
 *    MTLBuffers that we can access directly.
 *
 * 2. MTLComputePipelineState: A compiled Metal kernel ready for execution.
 *
 * 3. MTLCommandBuffer: A container for GPU commands. Commands are encoded,
 *    then committed for execution.
 *
 * 4. MTLComputeCommandEncoder: Used to encode compute commands (kernel
 * dispatches) into a command buffer.
 *
 * =============================================================================
 * PYTORCH MPS INTEGRATION
 * =============================================================================
 *
 * PyTorch provides helpers for MPS:
 * - torch::mps::get_command_buffer(): Get current command buffer
 * - torch::mps::commit(): Commit commands for execution
 * - torch::mps::synchronize(): Wait for GPU to finish
 *
 * To get the MTLBuffer from a PyTorch tensor:
 *   id<MTLBuffer> buffer = __bridge id<MTLBuffer>(tensor.storage().data());
 *
 * =============================================================================
 * MEMORY MANAGEMENT
 * =============================================================================
 *
 * - Input tensors: Use existing MTLBuffers from PyTorch tensors
 * - Output tensors: Create PyTorch tensors, use their MTLBuffers
 * - Temporary buffers: Can use PyTorch tensors or create MTLBuffers directly
 *
 * Important: Ensure all operations complete before returning to Python!
 * Use synchronize() or proper command buffer dependencies.
 */

#include "torch/mps.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <torch/extension.h>

// =============================================================================
// GLOBAL STATE
// =============================================================================

// Get the source directory of this file.
// Uses absolute path from Dl_info to handle different working directories.
#include <dlfcn.h>
NSString *getSourceDir() {
  Dl_info info;
  if (dladdr((void *)getSourceDir, &info)) {
    NSString *modulePath = [NSString stringWithUTF8String:info.dli_fname];
    // Module is at mps_ctc/_C.*.so, source is at mps_ctc/csrc/
    return [[modulePath stringByDeletingLastPathComponent]
        stringByAppendingPathComponent:@"csrc"];
  }
  // Fallback to compile-time path
  NSString *thisFile = @__FILE__;
  return [thisFile stringByDeletingLastPathComponent];
}

// Get the cached compiled metallib path.
NSString *getMetallibCacheDir() {
  NSString *cacheDir = [getSourceDir() stringByAppendingPathComponent:@"cache"];
  // Create cache dir if needed.
  [[NSFileManager defaultManager] createDirectoryAtPath:cacheDir
                            withIntermediateDirectories:YES
                                             attributes:nil
                                                  error:nil];
  return cacheDir;
}

// Get GPU architecture.
NSString *getGpuArchitecture(id<MTLDevice> device) {
  for (int family = 1020; family > 1001; family--) {
    if ([device supportsFamily:(MTLGPUFamily)family]) {
      return [NSString stringWithFormat:@"apple%d", family - 1000];
    }
  }
  return @"unknown";
}

/**
 * MetalContext: Singleton managing Metal resources.
 *
 * Holds the Metal device, library, and compiled kernel pipelines.
 * Initialized once on first use.
 */
struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> forward_kernel = nil;
  id<MTLComputePipelineState> backward_kernel = nil;
  id<MTLComputePipelineState> gradient_kernel = nil;
  id<MTLComputePipelineState> combined_kernel = nil;
  bool initialized = false;

  /**
   * Initialize Metal context.
   *
   * This function should:
   * 1. Get the default Metal device (MTLCreateSystemDefaultDevice)
   * 2. Load the compiled Metal library (.metallib file)
   * 3. Create compute pipeline states for each kernel
   *
   * The .metallib file is created by compiling ctc_kernels.metal:
   *   xcrun -sdk macosx metal -c ctc_kernels.metal -o ctc_kernels.air
   *   xcrun -sdk macosx metallib ctc_kernels.air -o ctc_kernels.metallib
   *
   * Or at runtime, compile from source using newLibraryWithSource.
   */
  void initialize() {
    if (initialized)
      return;

    @autoreleasepool {
      NSError *error = nil;

      // Step 1: Get Metal device
      device = MTLCreateSystemDefaultDevice();
      TORCH_CHECK(device != nil, "Failed to create Metal device");

      // Step 2: Load Metal library if exists otherwise recompile and save
      NSFileManager *fm = [NSFileManager defaultManager];

      NSString *arch = getGpuArchitecture(device);
      NSString *metallibPath = [getMetallibCacheDir()
          stringByAppendingPathComponent:[NSString
                                             stringWithFormat:@"%s%@%s",
                                                              "ctc_kernels_",
                                                              arch,
                                                              ".metallib"]];
      NSString *sourcePath =
          [getSourceDir() stringByAppendingPathComponent:@"ctc_kernels.metal"];
      TORCH_CHECK(
          [fm fileExistsAtPath:sourcePath],
          [NSString stringWithFormat:@"Source metal file does not exist at %@",
                                     sourcePath]);

      if ([fm fileExistsAtPath:metallibPath]) {
        NSDictionary *metallibAttrs = [fm attributesOfItemAtPath:metallibPath
                                                           error:nil];
        NSDictionary *sourceAttrs = [fm attributesOfItemAtPath:sourcePath
                                                         error:nil];

        NSDate *metallibDate = metallibAttrs[NSFileModificationDate];
        NSDate *sourceDate = sourceAttrs[NSFileModificationDate];

        BOOL isStale = [metallibDate compare:sourceDate] == NSOrderedAscending;
        if (!isStale) {
          error = nil;  // Reset error
          library = [device newLibraryWithFile:metallibPath error:&error];
          TORCH_CHECK(library != nil,
                      "Error when loading compiled ctc metal kernel: ",
                      error ? [[error localizedDescription] UTF8String] : "unknown error");
        }
      }

      if (library == nil) {
        error = nil;  // Reset error
        NSString *sourceString =
            [NSString stringWithContentsOfFile:sourcePath
                                      encoding:NSUTF8StringEncoding
                                         error:&error];
        TORCH_CHECK(sourceString != nil,
                    "Error when reading in source file of ctc metal kernel: ",
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
        error = nil;  // Reset error
        library = [device newLibraryWithSource:sourceString
                                       options:nil
                                         error:&error];
        TORCH_CHECK(library != nil,
                    "Error when compiling ctc metal kernel: ",
                    error ? [[error localizedDescription] UTF8String] : "unknown error");

        // Save compiled in background
        dispatch_async(
            dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0), ^{
              NSTask *task = [[NSTask alloc] init];
              task.executableURL = [NSURL fileURLWithPath:@"usr/bin/xcrun"];
              task.arguments = @[
                @"-sdk",
                @"macosx",
                @"metal",
                @"-o",
                metallibPath,
                sourcePath,
              ];

              NSError *error = nil;
              [task launchAndReturnError:&error];
              if (!error) {
                [task waitUntilExit];
                if (task.terminationStatus == 0) {
                  NSLog(@"Cached compiled metallib to : %@", metallibPath);
                }
              }
            });
      }

      // Step 3: Create pipeline states
      id<MTLFunction> forward_fn = [library newFunctionWithName:@"ctc_forward"];
      TORCH_CHECK(forward_fn != nil, "Failed to find ctc_forward function in Metal library");
      error = nil;
      forward_kernel = [device newComputePipelineStateWithFunction:forward_fn
                                                             error:&error];
      TORCH_CHECK(forward_kernel != nil,
                  "Failed to create forward kernel pipeline: ",
                  error ? [[error localizedDescription] UTF8String] : "unknown error");

      id<MTLFunction> backward_fn =
          [library newFunctionWithName:@"ctc_backward"];
      TORCH_CHECK(backward_fn != nil, "Failed to find ctc_backward function in Metal library");
      error = nil;
      backward_kernel = [device newComputePipelineStateWithFunction:backward_fn
                                                              error:&error];
      TORCH_CHECK(backward_kernel != nil,
                  "Failed to create backward kernel pipeline: ",
                  error ? [[error localizedDescription] UTF8String] : "unknown error");

      id<MTLFunction> gradient_fn =
          [library newFunctionWithName:@"ctc_gradient"];
      TORCH_CHECK(gradient_fn != nil, "Failed to find ctc_gradient function in Metal library");
      error = nil;
      gradient_kernel = [device newComputePipelineStateWithFunction:gradient_fn
                                                              error:&error];
      TORCH_CHECK(gradient_kernel != nil,
                  "Failed to create gradient kernel pipeline: ",
                  error ? [[error localizedDescription] UTF8String] : "unknown error");


      id<MTLFunction> combined_fn =
          [library newFunctionWithName:@"ctc_forward_backward_combined"];
      TORCH_CHECK(gradient_fn != nil, "Failed to find ctc_forward_backward_combined function in Metal library");
      error = nil;
      combined_kernel = [device newComputePipelineStateWithFunction:combined_fn
                                                              error:&error];
      TORCH_CHECK(combined_kernel != nil,
                  "Failed to create forward and backward combined kernel pipeline: ",
                  error ? [[error localizedDescription] UTF8String] : "unknown error");

      initialized = true;
    }
  }
};

// Global context instance
static MetalContext g_context;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Get MTLBuffer from PyTorch tensor.
 *
 * PyTorch MPS tensors store their data in MTLBuffers. This function
 * extracts the underlying buffer for use with Metal compute kernels.
 *
 * @param tensor PyTorch tensor (must be on MPS device and contiguous)
 * @return MTLBuffer containing the tensor data
 *
 * Usage:
 *   torch::Tensor t = torch::randn({10, 20}, torch::device(torch::kMPS));
 *   id<MTLBuffer> buf = getMTLBuffer(t);
 */
inline id<MTLBuffer> getMTLBuffer(const torch::Tensor &tensor) {
  // The tensor's storage data pointer is actually a void* that can be
  // bridged to an MTLBuffer
  return (__bridge id<MTLBuffer>)(tensor.storage().data());
}

/**
 * Get byte offset into MTLBuffer for tensor.
 *
 * A tensor may not start at the beginning of its storage buffer.
 * This returns the byte offset to the tensor's data.
 *
 * @param tensor PyTorch tensor
 * @return Byte offset into the MTLBuffer
 */
inline NSUInteger getMTLBufferOffset(const torch::Tensor &tensor) {
  return tensor.storage_offset() * tensor.element_size();
}

/**
 * Dispatch a compute kernel.
 *
 * Helper function to encode and dispatch a Metal compute kernel.
 *
 * @param encoder The compute command encoder
 * @param pipeline The compiled kernel pipeline
 * @param grid_size Total number of threads to launch
 * @param block_size Threads per threadgroup (usually 256 or less)
 */
inline void dispatchKernel(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline,
                           MTLSize grid_size, MTLSize block_size) {
  [encoder setComputePipelineState:pipeline];

  // Calculate number of threadgroups
  // Each dimension: ceil(grid / block)
  MTLSize threadgroups = MTLSizeMake(
      (grid_size.width + block_size.width - 1) / block_size.width,
      (grid_size.height + block_size.height - 1) / block_size.height,
      (grid_size.depth + block_size.depth - 1) / block_size.depth);

  [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:block_size];
}

// =============================================================================
// FORWARD PASS
// =============================================================================

/**
 * CTC Loss Forward Pass
 *
 * Computes CTC loss and forward variables (alpha) for each batch element.
 *
 * @param log_probs     Log probabilities [T, B, C] on MPS device
 *                      T = input sequence length
 *                      B = batch size
 *                      C = number of classes (including blank)
 *                      Should be output of log_softmax
 *
 * @param targets       Target sequences [B, S] on MPS device
 *                      S = max target sequence length
 *                      Values should be in [0, C-1], excluding blank for
 * non-blank positions
 *
 * @param input_lengths Length of each input sequence [B] on MPS device
 *                      Values should be in [1, T]
 *
 * @param target_lengths Length of each target sequence [B] on MPS device
 *                       Values should be in [1, S]
 *
 * @param blank         Index of blank label (usually 0)
 *
 * @return Tuple of:
 *         - loss: Per-sample losses [B]
 *         - alpha: Forward variables [T, B, L] where L = 2*S + 1
 *
 * The alpha tensor is returned for use in the backward pass.
 *
 * Example:
 *   auto [loss, alpha] = ctc_loss_forward(log_probs, targets, input_lengths,
 * target_lengths, 0);
 */
std::tuple<torch::Tensor, torch::Tensor>
ctc_loss_forward(const torch::Tensor &log_probs, const torch::Tensor &targets,
                 const torch::Tensor &input_lengths,
                 const torch::Tensor &target_lengths, int blank) {
  // Ensure Metal context is initialized
  g_context.initialize();

  // ==========================================================================
  // INPUT VALIDATION
  // ==========================================================================

  TORCH_CHECK(log_probs.device().is_mps(), "log_probs must be on MPS device");
  TORCH_CHECK(targets.device().is_mps(), "targets must be on MPS device");
  TORCH_CHECK(input_lengths.device().is_mps(),
              "input_lengths must be on MPS device");
  TORCH_CHECK(target_lengths.device().is_mps(),
              "target_lengths must be on MPS device");

  TORCH_CHECK(log_probs.dim() == 3, "log_probs must be 3D [T, B, C]");
  TORCH_CHECK(targets.dim() == 2, "targets must be 2D [B, S]");
  TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be 1D [B]");
  TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be 1D [B]");

  TORCH_CHECK(log_probs.is_contiguous(), "log_probs must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");

  // Get dimensions
  int T = log_probs.size(0); // Input sequence length
  int B = log_probs.size(1); // Batch size
  int C = log_probs.size(2); // Number of classes
  int S = targets.size(1);   // Max target length
  int L = 2 * S + 1;         // Expanded sequence length

  TORCH_CHECK(targets.size(0) == B, "targets batch size must match log_probs");
  TORCH_CHECK(input_lengths.size(0) == B,
              "input_lengths size must match batch size");
  TORCH_CHECK(target_lengths.size(0) == B,
              "target_lengths size must match batch size");

  TORCH_CHECK(blank >= 0 && blank < C, "blank must be in [0, C)");

  // ==========================================================================
  // ALLOCATE OUTPUT TENSORS
  // ==========================================================================

  // Loss tensor: one value per batch element
  auto loss = torch::empty({B}, log_probs.options());

  // Alpha tensor: forward variables
  // Shape: [T, B, L] where L = 2*S + 1
  auto alpha = torch::full({T, B, L}, -std::numeric_limits<float>::infinity(),
                           log_probs.options());

  // ==========================================================================
  // ENCODE METAL COMMANDS
  // ==========================================================================

  @autoreleasepool {
    // Step 1: Get command buffer from PyTorch MPS
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();

    // Step 2: Create compute command encoder
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];

    // Step 3: Set kernel and buffers
    [encoder setComputePipelineState:g_context.forward_kernel];
    [encoder setBuffer:getMTLBuffer(log_probs)
                offset:getMTLBufferOffset(log_probs)
               atIndex:0];
    [encoder setBuffer:getMTLBuffer(targets)
                offset:getMTLBufferOffset(targets)
               atIndex:1];
    [encoder setBuffer:getMTLBuffer(input_lengths)
                offset:getMTLBufferOffset(input_lengths)
               atIndex:2];
    [encoder setBuffer:getMTLBuffer(target_lengths)
                offset:getMTLBufferOffset(target_lengths)
               atIndex:3];
    [encoder setBuffer:getMTLBuffer(alpha)
                offset:getMTLBufferOffset(alpha)
               atIndex:4];
    [encoder setBuffer:getMTLBuffer(loss)
                offset:getMTLBufferOffset(loss)
               atIndex:5];

    // Step 4: Set scalar constants
    [encoder setBytes:&T length:sizeof(int) atIndex:6];
    [encoder setBytes:&B length:sizeof(int) atIndex:7];
    [encoder setBytes:&C length:sizeof(int) atIndex:8];
    [encoder setBytes:&S length:sizeof(int) atIndex:9];
    [encoder setBytes:&blank length:sizeof(int) atIndex:10];

    // Step 5: Dispatch kernel (one thread per batch element)
    MTLSize gridSize = MTLSizeMake(B, 1, 1);
    MTLSize blockSize = MTLSizeMake(fmin(B, 256), 1, 1);
    dispatchKernel(encoder, g_context.forward_kernel, gridSize, blockSize);
    
    // Step 6: End encoding
    [encoder endEncoding];
    
    // Step 7: Commit and synchronize (or use PyTorch's mechanism)
    torch::mps::commit();
    torch::mps::synchronize();
  }

  return {loss, alpha};
}

// =============================================================================
// BACKWARD PASS
// =============================================================================

/**
 * CTC Loss Backward Pass
 *
 * Computes gradients of the loss with respect to log_probs.
 *
 * This function:
 * 1. Computes backward variables (beta) using the backward kernel
 * 2. Computes gradients using alpha, beta, and the gradient kernel
 *
 * @param grad_output   Gradient of loss w.r.t. output [B] (from .backward())
 * @param log_probs     Original log probabilities [T, B, C]
 * @param alpha         Forward variables from forward pass [T, B, L]
 * @param targets       Target sequences [B, S]
 * @param input_lengths Input lengths [B]
 * @param target_lengths Target lengths [B]
 * @param blank         Blank label index
 *
 * @return Gradient w.r.t. log_probs [T, B, C]
 *
 * The gradient has the same shape as log_probs. Gradients for timesteps
 * beyond input_lengths[b] will be zero.
 */
torch::Tensor ctc_loss_backward(const torch::Tensor &grad_output,
                                const torch::Tensor &log_probs,
                                const torch::Tensor &alpha,
                                const torch::Tensor &targets,
                                const torch::Tensor &input_lengths,
                                const torch::Tensor &target_lengths,
                                int blank) {
  // Ensure context is initialized
  g_context.initialize();

  // Get dimensions
  int T = log_probs.size(0);
  int B = log_probs.size(1);
  int C = log_probs.size(2);
  int S = targets.size(1);
  int L = 2 * S + 1;

  // ==========================================================================
  // ALLOCATE TENSORS
  // ==========================================================================

  // Beta tensor: backward variables
  auto beta = torch::full({T, B, L}, -std::numeric_limits<float>::infinity(),
                          log_probs.options());

  // Gradient tensor
  auto grad = torch::zeros({T, B, C}, log_probs.options());

  // ==========================================================================
  // ENCODE METAL COMMANDS
  // ==========================================================================

  @autoreleasepool {
    // Part 1: Compute backward variables (beta)
    // -----------------------------------------
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:g_context.backward_kernel];
    [encoder setBuffer:getMTLBuffer(log_probs) offset:getMTLBufferOffset(log_probs) atIndex:0];
    [encoder setBuffer:getMTLBuffer(targets) offset:getMTLBufferOffset(targets) atIndex:1];
    [encoder setBuffer:getMTLBuffer(input_lengths) offset:getMTLBufferOffset(input_lengths) atIndex:2];
    [encoder setBuffer:getMTLBuffer(target_lengths) offset:getMTLBufferOffset(target_lengths) atIndex:3];
    [encoder setBuffer:getMTLBuffer(beta) offset:getMTLBufferOffset(beta) atIndex:4];
    [encoder setBytes:&T length:sizeof(int) atIndex:5];
    [encoder setBytes:&B length:sizeof(int) atIndex:6];
    [encoder setBytes:&C length:sizeof(int) atIndex:7];
    [encoder setBytes:&S length:sizeof(int) atIndex:8];
    [encoder setBytes:&blank length:sizeof(int) atIndex:9];

    // One thread per batch element
    MTLSize backwardGridSize = MTLSizeMake(B, 1, 1);
    MTLSize backwardBlockSize = MTLSizeMake(fmin(B, 256), 1, 1);
    dispatchKernel(encoder, g_context.backward_kernel, backwardGridSize, backwardBlockSize);
    [encoder endEncoding];
    
    // Part 2: Compute gradients
    // -------------------------
    encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:g_context.gradient_kernel];
    [encoder setBuffer:getMTLBuffer(grad_output) offset:getMTLBufferOffset(grad_output) atIndex:0];
    [encoder setBuffer:getMTLBuffer(log_probs) offset:getMTLBufferOffset(log_probs) atIndex:1];
    [encoder setBuffer:getMTLBuffer(alpha) offset:getMTLBufferOffset(alpha) atIndex:2];
    [encoder setBuffer:getMTLBuffer(beta) offset:getMTLBufferOffset(beta) atIndex:3];
    [encoder setBuffer:getMTLBuffer(targets) offset:getMTLBufferOffset(targets) atIndex:4];
    [encoder setBuffer:getMTLBuffer(input_lengths) offset:getMTLBufferOffset(input_lengths) atIndex:5];
    [encoder setBuffer:getMTLBuffer(target_lengths) offset:getMTLBufferOffset(target_lengths) atIndex:6];
    [encoder setBuffer:getMTLBuffer(grad) offset:getMTLBufferOffset(grad) atIndex:7];
    [encoder setBytes:&T length:sizeof(int) atIndex:8];
    [encoder setBytes:&B length:sizeof(int) atIndex:9];
    [encoder setBytes:&C length:sizeof(int) atIndex:10];
    [encoder setBytes:&S length:sizeof(int) atIndex:11];
    [encoder setBytes:&blank length:sizeof(int) atIndex:12];

    MTLSize gradientGridSize = MTLSizeMake(T, B, C);
    MTLSize gradientBlockSize = MTLSizeMake(8, 8, 4); // TODO: Tune for different architectures.
    dispatchKernel(encoder, g_context.gradient_kernel, gradientGridSize, gradientBlockSize);
    [encoder endEncoding];

    torch::mps::commit();
    torch::mps::synchronize();
  }

  return grad;
}

std::tuple<torch::Tensor, torch::Tensor>
ctc_loss_combined(const torch::Tensor &log_probs,
                  const torch::Tensor &targets,
                  const torch::Tensor &input_lengths,
                  const torch::Tensor &target_lengths,
                  const torch::Tensor &grad_output,
                  int blank) {
  g_context.initialize();

  // Dimensions
  int T = log_probs.size(0);
  int B = log_probs.size(1);
  int C = log_probs.size(2);
  int S = targets.size(1);

  // Allocate outputs
  auto loss = torch::empty({B}, log_probs.options());
  auto grad = torch::zeros({T, B, C}, log_probs.options());

  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:g_context.combined_kernel];
    [encoder setBuffer:getMTLBuffer(log_probs) offset:getMTLBufferOffset(log_probs) atIndex:0];
    [encoder setBuffer:getMTLBuffer(targets) offset:getMTLBufferOffset(targets) atIndex:1];
    [encoder setBuffer:getMTLBuffer(input_lengths) offset:getMTLBufferOffset(input_lengths) atIndex:2];
    [encoder setBuffer:getMTLBuffer(target_lengths) offset:getMTLBufferOffset(target_lengths) atIndex:3];
    [encoder setBuffer:getMTLBuffer(grad_output) offset:getMTLBufferOffset(grad_output) atIndex:4];
    [encoder setBuffer:getMTLBuffer(grad) offset:getMTLBufferOffset(grad) atIndex:5];
    [encoder setBuffer:getMTLBuffer(loss) offset:getMTLBufferOffset(loss) atIndex:6];
    [encoder setBytes:&T length:sizeof(int) atIndex:7];
    [encoder setBytes:&B length:sizeof(int) atIndex:8];
    [encoder setBytes:&C length:sizeof(int) atIndex:9];
    [encoder setBytes:&S length:sizeof(int) atIndex:10];
    [encoder setBytes:&blank length:sizeof(int) atIndex:11];

    MTLSize gridSize = MTLSizeMake(B, 1, 1);
    MTLSize blockSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];

    [encoder endEncoding];
    torch::mps::commit();
    torch::mps::synchronize();
  }

    return {loss, grad};
}


// =============================================================================
// PYTHON BINDINGS
// =============================================================================

/**
 * Register functions with PyTorch.
 *
 * This macro creates a Python module named mps_ctc._C with two functions:
 * - ctc_loss_forward
 * - ctc_loss_backward
 *
 * In Python:
 *   from mps_ctc import _C
 *   loss, alpha = _C.ctc_loss_forward(log_probs, targets, ...)
 *   grad = _C.ctc_loss_backward(grad_output, log_probs, alpha, ...)
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MPS-accelerated CTC Loss for PyTorch";

  m.def("ctc_loss_forward", &ctc_loss_forward, "CTC loss forward pass on MPS",
        py::arg("log_probs"), py::arg("targets"), py::arg("input_lengths"),
        py::arg("target_lengths"), py::arg("blank"));

  m.def("ctc_loss_backward", &ctc_loss_backward,
        "CTC loss backward pass on MPS", py::arg("grad_output"),
        py::arg("log_probs"), py::arg("alpha"), py::arg("targets"),
        py::arg("input_lengths"), py::arg("target_lengths"), py::arg("blank"));
  m.def("ctc_loss_combined", &ctc_loss_combined,
      "CTC loss combined forward-backward on MPS",
      py::arg("log_probs"), py::arg("targets"), py::arg("input_lengths"),
      py::arg("target_lengths"), py::arg("grad_output"), py::arg("blank"));

}
