/**
 * CTC Loss Metal Compute Kernels
 *
 * This file contains Metal compute shaders for computing CTC (Connectionist
 * Temporal Classification) loss and gradients on Apple Silicon GPUs.
 *
 * =============================================================================
 * CTC ALGORITHM OVERVIEW
 * =============================================================================
 *
 * CTC is used for sequence-to-sequence tasks where the alignment between input
 * and output is unknown (e.g., OCR, speech recognition).
 *
 * Key concepts:
 * 1. BLANK SYMBOL: A special symbol (usually index 0) that represents "no output"
 * 2. EXPANDED SEQUENCE: Target "cat" becomes [blank, c, blank, a, blank, t, blank]
 *    This allows the model to output blanks between characters and handles
 *    repeated characters (e.g., "book" needs blank between the two 'o's)
 * 3. FORWARD-BACKWARD ALGORITHM: Dynamic programming to sum over all valid alignments
 *
 * =============================================================================
 * MEMORY LAYOUT
 * =============================================================================
 *
 * All tensors use row-major (C-style) memory layout:
 * - log_probs[T, B, C]: Index = t * (B * C) + b * C + c
 * - alpha[T, B, L]:     Index = t * (B * L) + b * L + s
 * - targets[B, S]:      Index = b * S + s_idx
 *
 * Where:
 *   T = max input sequence length (time steps)
 *   B = batch size
 *   C = number of classes (vocabulary size including blank)
 *   S = max target sequence length
 *   L = expanded sequence length = 2 * S + 1
 *
 * =============================================================================
 * NUMERICAL STABILITY
 * =============================================================================
 *
 * All computations are done in LOG SPACE to prevent underflow.
 * Instead of: prob = prob_a * prob_b
 * We compute: log_prob = log_prob_a + log_prob_b
 *
 * For addition in log space: log(a + b) = log_sum_exp(log_a, log_b)
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Represents log(0) = -infinity in log space
constant float NEG_INF = -INFINITY;


// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Numerically stable log-sum-exp for two values.
 *
 * Computes: log(exp(a) + exp(b)) in a numerically stable way.
 *
 * The naive implementation exp(a) + exp(b) overflows/underflows for large/small values.
 * Instead, we use: max(a,b) + log(1 + exp(-|a-b|))
 *
 * Special cases:
 * - If a = -inf: return b
 * - If b = -inf: return a
 * - If both = -inf: return -inf
 *
 * @param a First log-probability
 * @param b Second log-probability
 * @return log(exp(a) + exp(b))
 *
 * Example:
 *   log_sum_exp(log(0.3), log(0.5)) = log(0.3 + 0.5) = log(0.8)
 */
inline float log_sum_exp(float a, float b) {
    // TODO: Implement numerically stable log-sum-exp
    //
    // Algorithm:
    // 1. Handle special case where a = -inf (return b)
    // 2. Handle special case where b = -inf (return a)
    // 3. Let max_val = max(a, b)
    // 4. Return max_val + log1p(exp(-abs(a - b)))
    //    (log1p computes log(1 + x) accurately for small x)

    return 0.0f; // PLACEHOLDER
}

/**
 * Log-sum-exp for three values.
 *
 * Computes: log(exp(a) + exp(b) + exp(c))
 *
 * @param a First log-probability
 * @param b Second log-probability
 * @param c Third log-probability
 * @return log(exp(a) + exp(b) + exp(c))
 */
inline float log_sum_exp3(float a, float b, float c) {
    // TODO: Implement using two calls to log_sum_exp
    return 0.0f; // PLACEHOLDER
}

/**
 * Get the label at position s in the expanded sequence.
 *
 * The expanded sequence alternates between blank and target characters:
 * Position: 0     1     2     3     4     5     6
 * Label:    blank t[0]  blank t[1]  blank t[2]  blank
 *
 * @param s Position in expanded sequence (0 to 2*S)
 * @param targets Pointer to target sequence for this batch element
 * @param blank Index of blank label
 * @return Label index at position s
 */
inline int get_label(int s, device const int* targets, int blank) {
    // TODO: Implement
    // If s is even: return blank
    // If s is odd: return targets[s / 2]
    return 0; // PLACEHOLDER
}


// =============================================================================
// FORWARD PASS KERNEL
// =============================================================================

/**
 * CTC Forward Pass Kernel
 *
 * Computes the forward variables (alpha) using dynamic programming.
 *
 * alpha[t, b, s] = log probability of outputting the first s symbols of the
 *                  expanded sequence given the first t frames of input.
 *
 * =============================================================================
 * ALGORITHM
 * =============================================================================
 *
 * Initialization (t = 0):
 *   alpha[0, b, 0] = log_probs[0, b, blank]        // Start with blank
 *   alpha[0, b, 1] = log_probs[0, b, targets[0]]   // Or start with first char
 *   alpha[0, b, s] = -inf for s > 1                // Can't skip ahead
 *
 * Recursion (t > 0):
 *   For each state s in expanded sequence:
 *     current_label = get_label(s)
 *
 *     // Can always stay in same state or come from previous state
 *     alpha[t, s] = log_sum_exp(alpha[t-1, s], alpha[t-1, s-1])
 *
 *     // Can skip a state if:
 *     //   - s > 1 (have a state to skip from)
 *     //   - current label is not blank
 *     //   - current label != label at s-2 (no skipping over repeated chars)
 *     if (s > 1 && current_label != blank && current_label != get_label(s-2)):
 *       alpha[t, s] = log_sum_exp(alpha[t, s], alpha[t-1, s-2])
 *
 *     // Add emission probability
 *     alpha[t, s] += log_probs[t, b, current_label]
 *
 * Final loss:
 *   loss[b] = -log_sum_exp(alpha[T-1, L-1], alpha[T-1, L-2])
 *   (Can end at final blank or final character)
 *
 * =============================================================================
 * PARALLELIZATION STRATEGY
 * =============================================================================
 *
 * Option 1: One thread per batch element (simple, less parallel)
 *   - Each thread handles full forward pass for one sequence
 *   - Good for small batches
 *
 * Option 2: Parallelize over states within timestep (complex, more parallel)
 *   - Synchronize between timesteps
 *   - Better for large sequences
 *
 * This skeleton uses Option 1 for simplicity.
 *
 * @param log_probs     Input log probabilities [T, B, C]
 * @param targets       Target sequences [B, S]
 * @param input_lengths Length of each input sequence [B]
 * @param target_lengths Length of each target sequence [B]
 * @param alpha         Output forward variables [T, B, L] where L = 2*max_S + 1
 * @param losses        Output per-sample losses [B]
 * @param T             Max input sequence length
 * @param B             Batch size
 * @param C             Number of classes
 * @param S             Max target sequence length
 * @param blank         Index of blank label
 * @param gid           Thread position in grid
 */
kernel void ctc_forward(
    device const float* log_probs       [[buffer(0)]],
    device const int*   targets         [[buffer(1)]],
    device const int*   input_lengths   [[buffer(2)]],
    device const int*   target_lengths  [[buffer(3)]],
    device float*       alpha           [[buffer(4)]],
    device float*       losses          [[buffer(5)]],
    constant int&       T               [[buffer(6)]],
    constant int&       B               [[buffer(7)]],
    constant int&       C               [[buffer(8)]],
    constant int&       S               [[buffer(9)]],
    constant int&       blank           [[buffer(10)]],
    uint                gid             [[thread_position_in_grid]]
) {
    // Each thread handles one batch element
    int b = gid;
    if (b >= B) return;

    // Get sequence lengths for this batch element
    int T_b = input_lengths[b];   // Input length for batch b
    int S_b = target_lengths[b];  // Target length for batch b
    int L_b = 2 * S_b + 1;        // Expanded sequence length

    // Maximum expanded sequence length (for indexing)
    int L = 2 * S + 1;

    // Pointer to this batch's targets
    device const int* targets_b = targets + b * S;

    // TODO: Implement forward pass
    //
    // Step 1: Initialize alpha at t=0
    //   - Set all alpha[0, b, :] to NEG_INF first
    //   - alpha[0, b, 0] = log_probs[0, b, blank]
    //   - if L_b > 1: alpha[0, b, 1] = log_probs[0, b, targets_b[0]]
    //
    // Step 2: Forward recursion for t = 1 to T_b - 1
    //   For each state s = 0 to L_b - 1:
    //     - Get current label using get_label(s, targets_b, blank)
    //     - Compute transitions (same state, previous state, skip state)
    //     - Add emission probability: log_probs[t, b, label]
    //     - Store in alpha[t, b, s]
    //
    // Step 3: Compute loss
    //   - Final alpha at last blank: alpha[T_b-1, b, L_b-1]
    //   - Final alpha at last char:  alpha[T_b-1, b, L_b-2]
    //   - loss[b] = -log_sum_exp(final_blank, final_char)
    //
    // Helper for indexing:
    //   alpha index = t * (B * L) + b * L + s
    //   log_probs index = t * (B * C) + b * C + c

    losses[b] = 0.0f; // PLACEHOLDER
}


// =============================================================================
// BACKWARD PASS KERNEL
// =============================================================================

/**
 * CTC Backward Pass Kernel
 *
 * Computes the backward variables (beta) using dynamic programming.
 *
 * beta[t, b, s] = log probability of outputting the remaining symbols of the
 *                 expanded sequence (from s to end) given frames t to T.
 *
 * =============================================================================
 * ALGORITHM
 * =============================================================================
 *
 * Initialization (t = T-1):
 *   beta[T-1, b, L-1] = 0  // log(1) = 0, can end at final blank
 *   beta[T-1, b, L-2] = 0  // Can also end at final character
 *   beta[T-1, b, s] = -inf for s < L-2
 *
 * Recursion (t < T-1), going backwards:
 *   For each state s:
 *     current_label = get_label(s)
 *
 *     // Transition TO same state or next state
 *     beta[t, s] = log_sum_exp(
 *       beta[t+1, s] + log_probs[t+1, current_label],     // stay
 *       beta[t+1, s+1] + log_probs[t+1, next_label]       // advance
 *     )
 *
 *     // Can skip if next-next label is different non-blank
 *     if (s+2 < L && get_label(s+2) != blank && get_label(s+2) != current_label):
 *       beta[t, s] = log_sum_exp(beta[t, s],
 *                                beta[t+1, s+2] + log_probs[t+1, get_label(s+2)])
 *
 * Note: The backward pass is symmetric to forward but runs in reverse.
 *
 * @param log_probs     Input log probabilities [T, B, C]
 * @param targets       Target sequences [B, S]
 * @param input_lengths Length of each input sequence [B]
 * @param target_lengths Length of each target sequence [B]
 * @param beta          Output backward variables [T, B, L]
 * @param T, B, C, S    Dimension constants
 * @param blank         Index of blank label
 * @param gid           Thread position
 */
kernel void ctc_backward(
    device const float* log_probs       [[buffer(0)]],
    device const int*   targets         [[buffer(1)]],
    device const int*   input_lengths   [[buffer(2)]],
    device const int*   target_lengths  [[buffer(3)]],
    device float*       beta            [[buffer(4)]],
    constant int&       T               [[buffer(5)]],
    constant int&       B               [[buffer(6)]],
    constant int&       C               [[buffer(7)]],
    constant int&       S               [[buffer(8)]],
    constant int&       blank           [[buffer(9)]],
    uint                gid             [[thread_position_in_grid]]
) {
    int b = gid;
    if (b >= B) return;

    int T_b = input_lengths[b];
    int S_b = target_lengths[b];
    int L_b = 2 * S_b + 1;
    int L = 2 * S + 1;

    device const int* targets_b = targets + b * S;

    // TODO: Implement backward pass
    //
    // Step 1: Initialize beta at t = T_b - 1
    //   - Set all beta[T_b-1, b, :] to NEG_INF
    //   - beta[T_b-1, b, L_b-1] = 0 (can end at final blank)
    //   - beta[T_b-1, b, L_b-2] = 0 (can end at final char)
    //
    // Step 2: Backward recursion for t = T_b - 2 down to 0
    //   For each state s = 0 to L_b - 1:
    //     - Compute transitions (same state, next state, skip state)
    //     - Note: transitions go FORWARD in state but BACKWARD in time
    //
    // The backward pass mirrors the forward pass structure but:
    //   - Iterates time backwards (T-2 down to 0)
    //   - Uses transitions to s, s+1, s+2 instead of from s, s-1, s-2
}


// =============================================================================
// GRADIENT KERNEL
// =============================================================================

/**
 * CTC Gradient Kernel
 *
 * Computes gradients of the loss with respect to log_probs.
 *
 * =============================================================================
 * ALGORITHM
 * =============================================================================
 *
 * The gradient is computed using alpha and beta:
 *
 * For each (t, b, c):
 *   grad[t, b, c] = sum over all s where label[s] == c of:
 *                   exp(alpha[t, s] + beta[t, s] - total_log_prob)
 *                   - exp(log_probs[t, b, c])
 *
 * Where total_log_prob = log_sum_exp(alpha[T-1, L-1], alpha[T-1, L-2])
 *                      = -loss[b]
 *
 * Intuition:
 * - alpha[t, s] + beta[t, s] = log prob of all paths passing through state s at time t
 * - We sum this over all states s that emit class c
 * - Subtract the predicted probability to get the gradient
 *
 * The gradient is then scaled by grad_output (from chain rule).
 *
 * @param grad_output   Incoming gradient [B] (from loss.backward())
 * @param log_probs     Input log probabilities [T, B, C]
 * @param alpha         Forward variables [T, B, L]
 * @param beta          Backward variables [T, B, L]
 * @param targets       Target sequences [B, S]
 * @param input_lengths Input lengths [B]
 * @param target_lengths Target lengths [B]
 * @param grad          Output gradient [T, B, C]
 * @param T, B, C, S    Dimensions
 * @param blank         Blank index
 * @param gid           Thread position (3D: t, b, c)
 */
kernel void ctc_gradient(
    device const float* grad_output     [[buffer(0)]],
    device const float* log_probs       [[buffer(1)]],
    device const float* alpha           [[buffer(2)]],
    device const float* beta            [[buffer(3)]],
    device const int*   targets         [[buffer(4)]],
    device const int*   input_lengths   [[buffer(5)]],
    device const int*   target_lengths  [[buffer(6)]],
    device float*       grad            [[buffer(7)]],
    constant int&       T               [[buffer(8)]],
    constant int&       B               [[buffer(9)]],
    constant int&       C               [[buffer(10)]],
    constant int&       S               [[buffer(11)]],
    constant int&       blank           [[buffer(12)]],
    uint3               gid             [[thread_position_in_grid]]
) {
    int t = gid.x;
    int b = gid.y;
    int c = gid.z;

    if (t >= T || b >= B || c >= C) return;

    int T_b = input_lengths[b];
    int S_b = target_lengths[b];
    int L_b = 2 * S_b + 1;
    int L = 2 * S + 1;

    // Skip if this timestep is beyond the sequence length
    if (t >= T_b) {
        grad[t * (B * C) + b * C + c] = 0.0f;
        return;
    }

    device const int* targets_b = targets + b * S;

    // TODO: Implement gradient computation
    //
    // Step 1: Get total log probability (= -loss[b])
    //   float total_log_prob = log_sum_exp(
    //     alpha[(T_b-1) * (B * L) + b * L + (L_b-1)],  // final blank
    //     alpha[(T_b-1) * (B * L) + b * L + (L_b-2)]   // final char
    //   );
    //
    // Step 2: Sum alpha + beta for all states that emit class c
    //   float log_grad = NEG_INF;
    //   for (int s = 0; s < L_b; s++) {
    //     if (get_label(s, targets_b, blank) == c) {
    //       float alpha_beta = alpha[t * (B * L) + b * L + s]
    //                        + beta[t * (B * L) + b * L + s];
    //       log_grad = log_sum_exp(log_grad, alpha_beta);
    //     }
    //   }
    //
    // Step 3: Compute gradient
    //   float prob = exp(log_probs[t * (B * C) + b * C + c]);
    //   float grad_val = exp(log_grad - total_log_prob) - prob;
    //
    // Step 4: Scale by grad_output and negate (since loss = -log_prob)
    //   grad[t * (B * C) + b * C + c] = -grad_val * grad_output[b];

    grad[t * (B * C) + b * C + c] = 0.0f; // PLACEHOLDER
}


// =============================================================================
// COMBINED FORWARD-BACKWARD KERNEL (OPTIONAL OPTIMIZATION)
// =============================================================================

/**
 * Combined Forward-Backward Kernel
 *
 * For better performance, you can combine forward and backward passes
 * into a single kernel to reduce memory bandwidth.
 *
 * This requires storing alpha in shared memory (threadgroup memory)
 * and computing beta on-the-fly while computing gradients.
 *
 * This is an advanced optimization - implement basic kernels first!
 */
// kernel void ctc_forward_backward_combined(...) { }
