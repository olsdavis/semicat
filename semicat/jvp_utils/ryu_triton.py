# this is copied (& modified) https://github.com/Ryu1845/min-sCM/blob/main/standalone_multihead_jvp_test.py#L326

# ruff: noqa: F841

import gc
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing


# --- Memory Tracking Utilities ---
def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    return 0.0


def get_peak_gpu_memory_usage():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def measure_memory_usage(fn, warmup=3, repetitions=10):
    """
    Measure peak memory usage of a function.

    Args:
        fn: Function to measure
        warmup: Number of warmup runs
        repetitions: Number of measurement runs

    Returns:
        tuple: (average_peak_memory_mb, max_peak_memory_mb)
    """
    peak_memories = []

    for _ in range(warmup):
        reset_gpu_memory_stats()
        fn()
        torch.cuda.synchronize()

    for _ in range(repetitions):
        reset_gpu_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak_memory = get_peak_gpu_memory_usage()
        peak_memories.append(peak_memory)

    avg_peak = sum(peak_memories) / len(peak_memories)
    max_peak = max(peak_memories)
    return avg_peak, max_peak


# --- Triton Multi-Head JVP Kernel ---


@triton.autotune(
    configs=[
        # Ultra-conservative configs for maximum compatibility
        # triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=1),
        # TODO @Mathias: fine tune
        # triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_warps=2, num_stages=1),
        # triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=1),
        # triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=1),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=4, num_stages=1),
        # triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    ],
    key=["B", "H", "L", "D_head"],
)
@triton.jit
def _flash_attention_jvp_multihead_kernel(
    # Input tensors
    Q,
    K,
    V,
    T_Q,
    T_K,
    T_V,
    # Output tensors
    Y,
    T_Y,
    M,
    MU,
    LI,
    # Tensor strides
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_tqb,
    stride_tqh,
    stride_tql,
    stride_tqd,
    stride_tkb,
    stride_tkh,
    stride_tkl,
    stride_tkd,
    stride_tvb,
    stride_tvh,
    stride_tvl,
    stride_tvd,
    stride_yb,
    stride_yh,
    stride_yl,
    stride_yd,
    stride_tyb,
    stride_tyh,
    stride_tyl,
    stride_tyd,
    # Problem dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    L_kv: tl.constexpr,
    L_div_up: tl.constexpr,
    D_head: tl.constexpr,
    D_head_pow2: tl.constexpr,
    # Scale factor
    scale: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention JVP kernel following the reference implementation pattern.
    Grid: (B*H, triton.cdiv(L, BLOCK_M))
    """
    # Get program IDs
    pid_bh = tl.program_id(0)  # Combined batch and head index
    pid_m = tl.program_id(1)  # Query block index

    # Decompose batch and head indices
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_head_pow2)

    # Base pointers for this (batch, head)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    tq_base = T_Q + pid_b * stride_tqb + pid_h * stride_tqh
    tk_base = T_K + pid_b * stride_tkb + pid_h * stride_tkh
    tv_base = T_V + pid_b * stride_tvb + pid_h * stride_tvh
    y_base = Y + pid_b * stride_yb + pid_h * stride_yh
    ty_base = T_Y + pid_b * stride_tyb + pid_h * stride_tyh

    # Load query block
    q_ptrs = q_base + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
    tq_ptrs = tq_base + offs_m[:, None] * stride_tql + offs_d[None, :] * stride_tqd

    mask_d = offs_d < D_head
    mask_m = offs_m < L
    mask_m_d = mask_m[:, None] & mask_d[None, :]

    q = tl.load(q_ptrs, mask=mask_m_d, other=0.0)
    tq = tl.load(tq_ptrs, mask=mask_m_d, other=0.0)

    # Initialize accumulators following Flash Attention pattern
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, D_head_pow2], dtype=tl.float32)
    g_acc = tl.zeros([BLOCK_M, D_head_pow2], dtype=tl.float32)
    mu_i = tl.zeros(
        [BLOCK_M], dtype=tl.float32
    )  # since we use alpha from m_i, we don't need to init to -inf
    p_tv_acc = tl.zeros([BLOCK_M, D_head_pow2], dtype=tl.float32)

    # Loop over key/value blocks
    for start_n in range(0, L_kv, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < L_kv
        mask_n_d = mask_n[:, None] & mask_d[None, :]

        # Load key and value blocks
        k_ptrs = k_base + offs_n_curr[:, None] * stride_kl + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n_curr[:, None] * stride_vl + offs_d[None, :] * stride_vd
        tk_ptrs = (
            tk_base + offs_n_curr[:, None] * stride_tkl + offs_d[None, :] * stride_tkd
        )
        tv_ptrs = (
            tv_base + offs_n_curr[:, None] * stride_tvl + offs_d[None, :] * stride_tvd
        )

        k = tl.load(k_ptrs, mask=mask_n_d, other=0.0)
        v = tl.load(v_ptrs, mask=mask_n_d, other=0.0)
        tk = tl.load(tk_ptrs, mask=mask_n_d, other=0.0)
        tv = tl.load(tv_ptrs, mask=mask_n_d, other=0.0)

        # Compute attention scores
        qk = tl.dot(q, tl.trans(k))  # mathias: a
        tqk = tl.dot(tq, tl.trans(k)) + tl.dot(q, tl.trans(tk))  # mathias: c

        # Mask invalid positions first
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        tqk = tl.where(mask_n[None, :], tqk, 0.0)

        # Online softmax computation following Flash Attention
        # m_ij = tl.maximum(m_i + tl.log(l_i), tl.max(qk, 1) * scale) # TODO @Mathias: this is how tridao does it. check what is more accurate
        m_ij = tl.maximum(m_i, tl.max(qk * scale, 1))  # this is how ryu does it.
        qk = qk * scale - m_ij[:, None]  # Scale and subtract max
        p = tl.math.exp(qk)  #                                              # mathias: r

        # Correction factor
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)  # mathias: k

        # Update normalization
        l_i = l_i * alpha + l_ij

        # Cast p back to input dtype for matmul
        p_typed = p.to(q.dtype)  # mathias: b? without l_i scale again

        # Update output accumulator
        acc = acc * alpha[:, None] + tl.dot(p_typed, v)

        # JVP accumulator: (p * tqk) @ v
        p_tqk = p * (
            tqk * scale  # mathias: f
        )  # Apply scale to tangent scores                              # mathias: j'ish (= r * f). but scaled by a factor of 1/4.5 - 1/7.0

        # Update mu: sum(p * tqk)
        mu_ij = tl.sum(p_tqk, 1)  # mathias: l
        mu_i = mu_i * alpha + mu_ij

        p_tqk_typed = p_tqk.to(q.dtype)  # Cast tangent weights too
        g_acc = g_acc * alpha[:, None] + tl.dot(
            p_tqk_typed, v
        )  # mathias: p? not exactly. unscaled

        # g_acc2 = g_acc * alpha[:, None] + tl.dot(o_unscaled.to(q.dtype), v)

        # Update p @ tv accumulator
        p_tv_acc = p_tv_acc * alpha[:, None] + tl.dot(
            p_typed, tv
        )  # mathias: q? without l_i scale again

        # Update max
        m_i = m_ij

    # Final computation - add log normalization and divide
    m_i += tl.math.log(l_i)
    y_out = acc / l_i[:, None]

    t_p_v = g_acc - mu_i[:, None] * y_out  # mathias:  p. (mu_i / l_i) = m
    t_y_out = (t_p_v + p_tv_acc) / l_i[:, None]

    # Store outputs
    y_ptrs = y_base + offs_m[:, None] * stride_yl + offs_d[None, :] * stride_yd
    ty_ptrs = ty_base + offs_m[:, None] * stride_tyl + offs_d[None, :] * stride_tyd

    tl.store(y_ptrs, y_out, mask=mask_m_d)
    tl.store(ty_ptrs, t_y_out, mask=mask_m_d)

    # Store max
    # m_ptrs = M + off_hz * N_CTX + offs_m
    m_ptrs = M + pid_bh * L_div_up + offs_m
    mu_ptrs = MU + pid_bh * L_div_up + offs_m
    li_ptrs = LI + pid_bh * L_div_up + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(mu_ptrs, mu_i / l_i)
    tl.store(li_ptrs, l_i)
    # tl.store(m_ptrs, lse_i)


def flash_attention_jvp_multihead_triton_kernel_wrapper(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    t_Q: torch.Tensor,
    t_K: torch.Tensor,
    t_V: torch.Tensor,
    scale: float | None = None,
    y: torch.Tensor | None = None,
    t_y: torch.Tensor | None = None,
    M: torch.Tensor | None = None,
    MU: torch.Tensor | None = None,
    LI: torch.Tensor | None = None,
    return_M: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Python wrapper for the Multi-head Flash Attention JVP Triton kernel.
    """
    device = Q.device
    dtype = Q.dtype
    B, H, L, D_head = Q.shape
    L_kv = K.shape[-2]

    # Check minimum dimension requirements for Triton
    if D_head < 16:
        raise ValueError(
            f"D_head must be >= 16 for efficient Triton kernel, got {D_head}"
        )

    if scale is None:
        scale = 1.0 / (D_head**0.5)

    # Ensure input shapes are correct
    assert Q.shape == (B, H, L, D_head), f"Q shape mismatch: {Q.shape}"
    assert K.shape == (B, H, L_kv, D_head), f"K shape mismatch: {K.shape}"
    assert V.shape == (B, H, L_kv, D_head), f"V shape mismatch: {V.shape}"
    assert t_Q.shape == (B, H, L, D_head), f"t_Q shape mismatch: {t_Q.shape}"
    assert t_K.shape == (B, H, L_kv, D_head), f"t_K shape mismatch: {t_K.shape}"
    assert t_V.shape == (B, H, L_kv, D_head), f"t_V shape mismatch: {t_V.shape}"

    # Create output tensors
    if y is None:
        y = torch.zeros((B, H, L, D_head), dtype=dtype, device=device)
    if t_y is None:
        t_y = torch.zeros((B, H, L, D_head), dtype=dtype, device=device)

    # Compute strides
    stride_qb, stride_qh, stride_ql, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_kl, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vl, stride_vd = V.stride()
    stride_tqb, stride_tqh, stride_tql, stride_tqd = t_Q.stride()
    stride_tkb, stride_tkh, stride_tkl, stride_tkd = t_K.stride()
    stride_tvb, stride_tvh, stride_tvl, stride_tvd = t_V.stride()
    stride_yb, stride_yh, stride_yl, stride_yd = y.stride()
    stride_tyb, stride_tyh, stride_tyl, stride_tyd = t_y.stride()

    L_div_up = triton.cdiv(L, 128) * 128

    if M is None:
        M = torch.empty((B, H, L_div_up), device=Q.device, dtype=torch.float32)
    if MU is None:
        MU = torch.empty((B, H, L_div_up), device=Q.device, dtype=torch.float32)
    if LI is None:
        LI = torch.empty((B, H, L_div_up), device=Q.device, dtype=torch.float32)

    D_head_pow2 = 2 ** math.ceil(math.log2(D_head))
    # Use block-based grid like Flash Attention
    # Choose BLOCK_M based on autotuning, but ensure we cover all queries
    # BLOCK_M = 64  # Will be determined by autotuning
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_M"]))

    _flash_attention_jvp_multihead_kernel[grid](
        Q,
        K,
        V,
        t_Q,
        t_K,
        t_V,
        y,
        t_y,
        M,
        MU,
        LI,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_tqb,
        stride_tqh,
        stride_tql,
        stride_tqd,
        stride_tkb,
        stride_tkh,
        stride_tkl,
        stride_tkd,
        stride_tvb,
        stride_tvh,
        stride_tvl,
        stride_tvd,
        stride_yb,
        stride_yh,
        stride_yl,
        stride_yd,
        stride_tyb,
        stride_tyh,
        stride_tyl,
        stride_tyd,
        B,
        H,
        L,
        L_kv,
        L_div_up,
        D_head,
        D_head_pow2,
        scale,
    )
    if return_M:
        return y, t_y, M, MU, LI
    else:
        return y, t_y


# --- Naive PyTorch Multi-Head JVP ---
def naive_multihead_attention_jvp(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    t_Q: torch.Tensor,
    t_K: torch.Tensor,
    t_V: torch.Tensor,
    scale: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive implementation of multi-head attention JVP using torch.func.jvp.
    """
    from torch.func import (
        jvp,  # Import locally to avoid issues if torch.func is not available
    )

    B, H, L, D_head = Q.shape

    if scale is None:
        scale = 1.0 / (D_head**0.5)

    def multihead_attention_forward(Q_param, K_param, V_param):
        scores = torch.matmul(Q_param, K_param.transpose(-2, -1)) * scale
        p = F.softmax(scores, dim=-1)
        y = torch.matmul(p, V_param)
        return y

    y, t_y = jvp(multihead_attention_forward, (Q, K, V), (t_Q, t_K, t_V))
    return y, t_y


# --- PyTorch Scaled Dot Product Attention Wrapper ---
def pytorch_scaled_dot_product_attention_wrapper(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float = None,  # Flash Attn CUDA handles scale internally if None
) -> torch.Tensor:
    """
    Wrapper for torch.nn.functional.scaled_dot_product_attention.
    Note: JVP is not supported by this function.
    """
    # from kernels import get_kernel
    # flash_attn = get_kernel("kernels-community/flash-attn")
    # B, H, L, D_head = Q.shape
    # scaled_dot_product_attention expects (B, H, L, D_head)
    # or (B*H, L, D_head) if we reshape. For multi-head, (B,H,L,D) is fine.

    # The 'scale' argument in F.scaled_dot_product_attention is a keyword-only argument.
    # If scale is None, F.scaled_dot_product_attention defaults to 1/sqrt(D_head).
    # If a scale value is provided, it will be used.
    # We ensure dropout_p is 0.0 for a fair comparison of the forward pass.
    Q.grad = None
    K.grad = None
    V.grad = None
    with torch.nn.attention.sdpa_kernel(
        backends=[torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
    ):
        o = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
        )
        dO = torch.empty_like(o)
        torch.autograd.backward(o, dO)

    # return flash_attn.mha_fwd(Q.transpose(1, 2).half(), K.transpose(1, 2).half(), V.transpose(1, 2).half(), is_causal=False, p_dropout=0.0, softmax_scale=scale)[0].to(Q.dtype).transpose(1, 2)
    return o


# --- Test Function ---
def test_multihead_correctness():
    """
    Tests the Triton multi-head JVP kernel against the naive PyTorch implementation.
    """
    print("Running Multi-Head JVP Correctness Test...")

    torch.manual_seed(42)
    device = torch.device("cuda")

    test_configs = [
        {"B": 1, "H": 1, "L": 8, "D_head": 16},  # Basic
        {"B": 2, "H": 2, "L": 16, "D_head": 32},  # Larger dimensions
        {"B": 1, "H": 4, "L": 32, "D_head": 32},  # More heads
        {"B": 1, "H": 1, "L": 64, "D_head": 64},  # Larger L, D_head
        {"B": 2, "H": 3, "L": 128, "D_head": 16},  # Mixed dimensions
    ]

    for i, config in enumerate(test_configs):
        B, H, L, D_head = config["B"], config["H"], config["L"], config["D_head"]

        print(f"\nTest Case {i+1}: B={B}, H={H}, L={L}, D_head={D_head}")

        Q = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        K = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        V = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        t_Q = (
            torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1
        )  # Smaller tangents
        t_K = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1
        t_V = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1

        # For consistency in testing, let's explicitly pass the scale.
        explicit_scale = 1.0 / (D_head**0.5)

        # Compute using naive PyTorch implementation
        y_naive, t_y_naive = naive_multihead_attention_jvp(
            Q, K, V, t_Q, t_K, t_V, explicit_scale
        )

        # Compute using Flash Attn CUDA (FlashAttention kernel)
        y_flashattn_cuda = pytorch_scaled_dot_product_attention_wrapper(
            Q, K, V, explicit_scale
        )

        # Compute using Triton kernel implementation
        try:
            y_triton, t_y_triton = flash_attention_jvp_multihead_triton_kernel_wrapper(
                Q, K, V, t_Q, t_K, t_V, explicit_scale
            )
        except Exception as e:
            print(f"  Triton kernel execution failed: {e}")
            # If triton fails, we can't compare. Mark as failure for this config.
            assert False, f"Triton kernel failed for config {config}"
            continue

        # Compare results
        rtol, atol = (
            1e-2,
            1e-2,
        )  # More realistic tolerances for Triton kernel differences

        try:
            torch.testing.assert_close(
                y_triton,
                y_naive,
                rtol=rtol,
                atol=atol,
                msg=f"Forward output (Triton vs Naive) mismatch for config {config}",
            )
            print("  Forward output (Triton vs Naive): PASSED")
        except AssertionError as e:
            print(f"  Forward output (Triton vs Naive): FAILED\n{e}")

        try:
            torch.testing.assert_close(
                y_triton,
                y_flashattn_cuda,
                rtol=rtol,
                atol=atol,
                msg=f"Forward output (Triton vs Flash Attn CUDA) mismatch for config {config}",
            )
            print("  Forward output (Triton vs Flash Attn CUDA): PASSED")
        except AssertionError as e:
            print(f"  Forward output (Triton vs Flash Attn CUDA): FAILED\n{e}")

        try:
            torch.testing.assert_close(
                t_y_triton,
                t_y_naive,
                rtol=rtol,
                atol=atol,
                msg=f"JVP output mismatch for config {config}",
            )
            print("  JVP output: PASSED")
        except AssertionError as e:
            print(f"  JVP output: FAILED\n{e}")

    print("\nMulti-Head JVP Correctness Test Finished.")


# --- Memory Test Function ---
def test_memory_usage():
    """
    Tests and compares memory usage between Triton and naive implementations.
    """
    print("Running Memory Usage Comparison...")

    torch.manual_seed(42)
    device = torch.device("cuda")

    test_configs = [
        {"B": 1, "H": 1, "L": 128, "D_head": 64},
        {"B": 2, "H": 2, "L": 256, "D_head": 32},
        {"B": 1, "H": 4, "L": 512, "D_head": 64},
    ]

    for i, config in enumerate(test_configs):
        B, H, L, D_head = config["B"], config["H"], config["L"], config["D_head"]

        print(f"\nMemory Test {i+1}: B={B}, H={H}, L={L}, D_head={D_head}")

        Q = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        K = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        V = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        t_Q = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1
        t_K = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1
        t_V = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1

        explicit_scale = 1.0 / (D_head**0.5)

        # Test Triton implementation memory usage
        triton_fn = lambda: flash_attention_jvp_multihead_triton_kernel_wrapper(
            Q, K, V, t_Q, t_K, t_V, explicit_scale
        )
        triton_avg_mem, triton_max_mem = measure_memory_usage(triton_fn)

        # Test naive implementation memory usage
        naive_fn = lambda: naive_multihead_attention_jvp(
            Q, K, V, t_Q, t_K, t_V, explicit_scale
        )
        naive_avg_mem, naive_max_mem = measure_memory_usage(naive_fn)

        # Test Flash Attn CUDA memory usage (forward pass only)
        flashattn_cuda_fn = lambda: pytorch_scaled_dot_product_attention_wrapper(
            Q, K, V, explicit_scale
        )
        flashattn_cuda_avg_mem, flashattn_cuda_max_mem = measure_memory_usage(
            flashattn_cuda_fn
        )

        print(
            f"  Triton       - Avg: {triton_avg_mem:.2f} MB, Peak: {triton_max_mem:.2f} MB (JVP)"
        )
        print(
            f"  Naive        - Avg: {naive_avg_mem:.2f} MB, Peak: {naive_max_mem:.2f} MB (JVP)"
        )
        print(
            f"  Flash Attn CUDA - Avg: {flashattn_cuda_avg_mem:.2f} MB, Peak: {flashattn_cuda_max_mem:.2f} MB (Forward Only)"
        )

        if triton_avg_mem > 0:
            memory_ratio_triton_vs_naive = naive_avg_mem / triton_avg_mem
            print(
                f"  Memory Efficiency (Naive JVP / Triton JVP): {memory_ratio_triton_vs_naive:.2f}x"
            )
        if (
            flashattn_cuda_avg_mem > 0 and triton_avg_mem > 0
        ):  # Compare forward pass memory if desired
            # Note: Triton JVP includes memory for tangents, Flash Attn CUDA is forward only
            # A direct comparison of Triton JVP vs Flash Attn CUDA forward memory is not apples-to-apples
            # but we can report the numbers.
            pass

    print("\nMemory Usage Comparison Finished.")


# --- Comprehensive Memory Analysis ---
def comprehensive_memory_analysis():
    """
    Comprehensive memory analysis comparing Triton vs Naive implementations
    across different problem sizes with detailed statistics.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MEMORY ANALYSIS")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Extended test configurations for comprehensive analysis
    test_configs = [
        # Small problems
        {"B": 1, "H": 1, "L": 64, "D_head": 32, "category": "Small"},
        {"B": 1, "H": 2, "L": 128, "D_head": 64, "category": "Small"},
        # Medium problems
        {"B": 2, "H": 4, "L": 256, "D_head": 64, "category": "Medium"},
        {"B": 1, "H": 8, "L": 512, "D_head": 32, "category": "Medium"},
        # Large problems
        {"B": 4, "H": 4, "L": 512, "D_head": 64, "category": "Large"},
        {"B": 2, "H": 8, "L": 1024, "D_head": 32, "category": "Large"},
    ]

    memory_results = []

    for i, config in enumerate(test_configs):
        B, H, L, D_head = config["B"], config["H"], config["L"], config["D_head"]
        category = config["category"]

        print(
            f"\n[{category.upper()}] Test {i+1}: B={B}, H={H}, L={L}, D_head={D_head}"
        )

        # Calculate theoretical memory requirements
        input_size = B * H * L * D_head * 2  # 2 bytes per float16
        total_input_size = input_size * 6  # Q, K, V, t_Q, t_K, t_V
        output_size = input_size * 2  # Y, t_Y

        print(f"  Theoretical Input Memory: {total_input_size / (1024*1024):.2f} MB")
        print(f"  Theoretical Output Memory: {output_size / (1024*1024):.2f} MB")

        Q = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        K = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        V = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16)
        t_Q = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1
        t_K = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1
        t_V = torch.randn(B, H, L, D_head, device=device, dtype=torch.float16) * 0.1

        explicit_scale = 1.0 / (D_head**0.5)

        # Test Triton implementation
        triton_fn = lambda: flash_attention_jvp_multihead_triton_kernel_wrapper(
            Q, K, V, t_Q, t_K, t_V, explicit_scale
        )
        triton_avg_mem, triton_peak_mem = measure_memory_usage(
            triton_fn, repetitions=10
        )

        # Test naive implementation
        naive_fn = lambda: naive_multihead_attention_jvp(
            Q, K, V, t_Q, t_K, t_V, explicit_scale
        )
        naive_avg_mem, naive_peak_mem = measure_memory_usage(naive_fn, repetitions=10)

        # Test Flash Attn CUDA memory usage (forward pass only)
        flashattn_cuda_fn = lambda: pytorch_scaled_dot_product_attention_wrapper(
            Q, K, V, explicit_scale
        )
        flashattn_cuda_avg_mem, flashattn_cuda_peak_mem = measure_memory_usage(
            flashattn_cuda_fn, repetitions=10
        )

        # Calculate efficiency metrics
        # Triton JVP vs Naive JVP
        memory_ratio_jvp = (
            naive_avg_mem / triton_avg_mem if triton_avg_mem > 0 else float("inf")
        )
        peak_ratio_jvp = (
            naive_peak_mem / triton_peak_mem if triton_peak_mem > 0 else float("inf")
        )
        memory_savings_jvp = naive_avg_mem - triton_avg_mem

        # Store results
        result = {
            "config": config,
            "triton_avg": triton_avg_mem,
            "triton_peak": triton_peak_mem,
            "naive_avg": naive_avg_mem,
            "naive_peak": naive_peak_mem,
            "flashattn_cuda_avg": flashattn_cuda_avg_mem,
            "flashattn_cuda_peak": flashattn_cuda_peak_mem,
            "ratio_jvp": memory_ratio_jvp,  # Naive JVP / Triton JVP
            "peak_ratio_jvp": peak_ratio_jvp,
            "savings_jvp": memory_savings_jvp,
        }
        memory_results.append(result)

        print(
            f"  Triton JVP     - Avg: {triton_avg_mem:8.2f} MB, Peak: {triton_peak_mem:8.2f} MB"
        )
        print(
            f"  Naive JVP      - Avg: {naive_avg_mem:8.2f} MB, Peak: {naive_peak_mem:8.2f} MB"
        )
        print(
            f"  Flash Attn CUDA - Avg: {flashattn_cuda_avg_mem:8.2f} MB, Peak: {flashattn_cuda_peak_mem:8.2f} MB (Forward Only)"
        )
        print(
            f"  Efficiency Gain (JVP Naive/Triton): {memory_ratio_jvp:.2f}x (avg), {peak_ratio_jvp:.2f}x (peak)"
        )
        print(f"  Memory Saved (JVP Triton vs Naive): {memory_savings_jvp:.2f} MB")

    # Summary statistics
    print("\n" + "=" * 60)
    print("MEMORY ANALYSIS SUMMARY")
    print("=" * 60)

    avg_ratios_jvp = [
        r["ratio_jvp"] for r in memory_results if r["ratio_jvp"] != float("inf")
    ]
    peak_ratios_jvp = [
        r["peak_ratio_jvp"]
        for r in memory_results
        if r["peak_ratio_jvp"] != float("inf")
    ]
    total_savings_jvp = sum(r["savings_jvp"] for r in memory_results)

    # Averages for Flash Attn CUDA (Forward only)
    avg_flashattn_cuda_mem = (
        sum(r["flashattn_cuda_avg"] for r in memory_results) / len(memory_results)
        if memory_results
        else 0
    )

    print(
        f"Average Memory Efficiency (Naive JVP / Triton JVP): {sum(avg_ratios_jvp)/len(avg_ratios_jvp):.2f}x"
        if avg_ratios_jvp
        else "N/A"
    )
    print(
        f"Peak Memory Efficiency (Naive JVP / Triton JVP): {sum(peak_ratios_jvp)/len(peak_ratios_jvp):.2f}x"
        if peak_ratios_jvp
        else "N/A"
    )
    print(f"Total Memory Saved (Triton JVP vs Naive JVP): {total_savings_jvp:.2f} MB")
    print(
        f"Average Flash Attn CUDA Forward Pass Memory: {avg_flashattn_cuda_mem:.2f} MB"
    )

    # Category breakdown for JVP comparison
    categories_jvp = {}
    for result in memory_results:
        cat = result["config"]["category"]
        if cat not in categories_jvp:
            categories_jvp[cat] = []
        categories_jvp[cat].append(result["ratio_jvp"])

    print("\nEfficiency (Naive JVP / Triton JVP) by Problem Size:")
    for cat, ratios in categories_jvp.items():
        valid_ratios = [r for r in ratios if r != float("inf")]
        if valid_ratios:
            avg_ratio = sum(valid_ratios) / len(valid_ratios)
            print(f"  {cat}: {avg_ratio:.2f}x average efficiency")
        else:
            print(f"  {cat}: N/A average efficiency")

    print("\n" + "=" * 60)


# --- Benchmark Function and Configurations ---

# Benchmark configurations and function are now defined unconditionally
benchmark_configs = [
    triton.testing.Benchmark(
        x_names=["L"],
        x_vals=[
            2**i for i in range(5, 17)
        ],  # Sequence lengths: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
        line_arg="provider",
        # line_vals=['triton', 'naive', 'flashattn_cuda'],
        line_vals=["triton", "flashattn_cuda"],
        # line_names=['Triton Kernel', 'Naive PyTorch', 'Flash Attn CUDA'],
        line_names=["Triton Kernel", "Flash Attn CUDA"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="ms",
        plot_name="multihead_jvp_L_scaling-B1-H2-D32-fp16",
        args={"B": 1, "H": 24, "D_head": 128, "dtype": torch.float16, "device": "cuda"},
    ),
    triton.testing.Benchmark(
        x_names=["L"],
        x_vals=[2**i for i in range(5, 17)],
        line_arg="provider",
        # line_vals=['triton', 'naive', 'flashattn_cuda'],
        line_vals=["triton", "flashattn_cuda"],
        # line_names=['Triton Kernel', 'Naive PyTorch', 'Flash Attn CUDA'],
        line_names=["Triton Kernel", "Flash Attn CUDA"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="TFLOP/s",
        plot_name="multihead_jvp_tflops_L_scaling-B1-H2-D32-fp16",
        args={
            "B": 1,
            "H": 24,
            "D_head": 128,
            "dtype": torch.float16,
            "device": "cuda",
            "benchmark_type": "tflops",
        },
    ),
    triton.testing.Benchmark(
        x_names=["D_head"],
        # x_vals=[16, 32, 64, 128, 256], # Head dimensions (all power-of-2)
        x_vals=[16, 32, 64, 128],  # Head dimensions (all power-of-2)
        line_arg="provider",
        # line_vals=['triton', 'naive', 'flashattn_cuda'],
        line_vals=["triton", "flashattn_cuda"],
        # line_names=['Triton Kernel', 'Naive PyTorch', 'Flash Attn CUDA'],
        line_names=["Triton Kernel", "Flash Attn CUDA"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="ms",
        plot_name="multihead_jvp_D_head_scaling-B1-H2-L1024-fp16",
        args={"B": 1, "H": 24, "L": 1024 * 8, "dtype": torch.float16, "device": "cuda"},
    ),
]

# Memory benchmark configurations
memory_benchmark_configs = [
    triton.testing.Benchmark(
        x_names=["L"],
        x_vals=[
            2**i for i in range(5, 13)
        ],  # Sequence lengths: 32, 64, 128, 256, 512, 1024, 2048, 4096
        line_arg="provider",
        line_vals=["triton", "naive", "flashattn_cuda"],
        line_names=["Triton Kernel", "Naive PyTorch", "Flash Attn CUDA"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Memory Usage (MB)",
        plot_name="multihead_jvp_memory_L_scaling-B1-H2-D32-fp16",
        args={
            "B": 1,
            "H": 2,
            "D_head": 32,
            "dtype": torch.float16,
            "device": "cuda",
            "benchmark_type": "memory",
        },
    ),
    triton.testing.Benchmark(
        x_names=["D_head"],
        x_vals=[16, 32, 64, 128, 256],  # Head dimensions
        line_arg="provider",
        line_vals=["triton", "naive", "flashattn_cuda"],
        line_names=["Triton Kernel", "Naive PyTorch", "Flash Attn CUDA"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Memory Usage (MB)",
        plot_name="multihead_jvp_memory_D_head_scaling-B1-H2-L512-fp16",
        args={
            "B": 1,
            "H": 2,
            "L": 512,
            "dtype": torch.float16,
            "device": "cuda",
            "benchmark_type": "memory",
        },
    ),
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[1, 2, 4, 8, 16],  # Batch sizes
        line_arg="provider",
        line_vals=["triton", "naive", "flashattn_cuda"],
        line_names=["Triton Kernel", "Naive PyTorch", "Flash Attn CUDA"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Memory Usage (MB)",
        plot_name="multihead_jvp_memory_B_scaling-H2-L256-D32-fp16",
        args={
            "H": 2,
            "L": 256,
            "D_head": 32,
            "dtype": torch.float16,
            "device": "cuda",
            "benchmark_type": "memory",
        },
    ),
]


@triton.testing.perf_report(benchmark_configs)
def bench_multihead_jvp(
    B, H, L, D_head, provider, dtype, device, benchmark_type="time"
):
    torch.manual_seed(0)  # Ensure consistent inputs for fair comparison

    Q = torch.randn(B, H, L, D_head, device=device, dtype=dtype)
    K = torch.randn(B, H, L, D_head, device=device, dtype=dtype)
    V = torch.randn(B, H, L, D_head, device=device, dtype=dtype)
    t_Q = torch.randn(B, H, L, D_head, device=device, dtype=dtype) * 0.1
    t_K = torch.randn(B, H, L, D_head, device=device, dtype=dtype) * 0.1
    t_V = torch.randn(B, H, L, D_head, device=device, dtype=dtype) * 0.1

    scale = 1.0 / (D_head**0.5) if D_head > 0 else 1.0

    if provider == "triton":
        fn = lambda: flash_attention_jvp_multihead_triton_kernel_wrapper(
            Q, K, V, t_Q, t_K, t_V, scale
        )
    elif provider == "naive":
        fn = lambda: naive_multihead_attention_jvp(Q, K, V, t_Q, t_K, t_V, scale)
    elif provider == "flashattn_cuda":
        # For flashattn_cuda, we only benchmark the forward pass
        Q_rg, K_rg, V_rg = (
            Q.clone().requires_grad_(True),
            K.clone().requires_grad_(True),
            V.clone().requires_grad_(True),
        )
        fn = lambda: pytorch_scaled_dot_product_attention_wrapper(
            Q_rg, K_rg, V_rg, scale
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if benchmark_type == "memory":
        # Measure memory usage
        avg_memory, max_memory = measure_memory_usage(fn, warmup=3, repetitions=5)
        return avg_memory
    else:
        # Measure time (default)
        ms = triton.testing.do_bench(fn, warmup=10, rep=50)
        if benchmark_type == "tflops":
            # JVP flop count: 10 * B * H * L^2 * D_head
            # Forward-only flop count: 4 * B * H * L^2 * D_head
            if provider in ["triton", "naive"]:
                flops = 10 * B * H * L**2 * D_head
            else:  # flashattn_cuda
                flops = 4 * B * H * L**2 * D_head
            return flops / ms / 1e9  # TFLOP/s
        return ms


# Memory-specific benchmark function
@triton.testing.perf_report(memory_benchmark_configs)
def bench_multihead_jvp_memory(
    B, H, L, D_head, provider, dtype, device, benchmark_type="memory"
):
    torch.manual_seed(0)  # Ensure consistent inputs for fair comparison

    Q = torch.randn(B, H, L, D_head, device=device, dtype=dtype)
    K = torch.randn(B, H, L, D_head, device=device, dtype=dtype)
    V = torch.randn(B, H, L, D_head, device=device, dtype=dtype)
    t_Q = torch.randn(B, H, L, D_head, device=device, dtype=dtype) * 0.1
    t_K = torch.randn(B, H, L, D_head, device=device, dtype=dtype) * 0.1
    t_V = torch.randn(B, H, L, D_head, device=device, dtype=dtype) * 0.1

    scale = 1.0 / (D_head**0.5) if D_head > 0 else 1.0

    if provider == "triton":
        fn = lambda: flash_attention_jvp_multihead_triton_kernel_wrapper(
            Q, K, V, t_Q, t_K, t_V, scale
        )
    elif provider == "naive":
        fn = lambda: naive_multihead_attention_jvp(Q, K, V, t_Q, t_K, t_V, scale)
    elif provider == "flashattn_cuda":
        # For flashattn_cuda, we only benchmark the forward pass memory
        fn = lambda: pytorch_scaled_dot_product_attention_wrapper(Q, K, V, scale)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Measure memory usage
    avg_memory, max_memory = measure_memory_usage(fn, warmup=3, repetitions=5)
    return avg_memory


if __name__ == "__main__":
    # test_multihead_correctness()

    # print("\nRunning Memory Usage Tests...")
    # test_memory_usage()

    # print("\nRunning Comprehensive Memory Analysis...")
    # comprehensive_memory_analysis()

    # Benchmarks are now run unconditionally
    print("\nRunning Multi-Head JVP Time Benchmarks...")
    bench_multihead_jvp.run(print_data=True, save_path=".")

    # print("\nRunning Multi-Head JVP Memory Benchmarks...")
    # bench_multihead_jvp_memory.run(print_data=True, save_path='.')

    # print("\nAll benchmarks finished. Plots saved to current directory.")
