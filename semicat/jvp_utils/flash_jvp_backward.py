"""
This is a modified version of the flash_attn_triton.py file.

This file includes several kernels that together process the 6 gradients of the JVP of an SDPA.
The kernels are split up into multiple smaller ones to lower the register pressure and therefore also the number of local memory transfers.
DQ and DTQ are processed separately to DK, DV, DTK, DTV, because atomics proved to be much slower than having separate kernels doing duplicate work,
but in well parallelizable, coalesced manner.


NOTE: variable names are mostly coming from torch compiler's jvp sdpa backward pass.
keeping the same variable names for debugging purposes.

"""
# ruff: noqa: F841

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_store_tensor(
    ptr,
    vals,
    offs_dim0,
    offs_dim1,
    max_dim0,
    max_dim1,
    EVEN_DIM0: tl.constexpr,
    EVEN_DIM1: tl.constexpr,
):
    if EVEN_DIM0:
        if EVEN_DIM1:
            tl.store(ptr, vals)
        else:
            tl.store(ptr, vals, mask=offs_dim1[None, :] < max_dim1)
    else:
        if EVEN_DIM1:
            tl.store(ptr, vals, mask=offs_dim0[:, None] < max_dim0)
        else:
            tl.store(
                ptr,
                vals,
                mask=(offs_dim0[:, None] < max_dim0) & (offs_dim1[None, :] < max_dim1),
            )


@triton.jit
def _bwd_store_2_tensors(
    a_ptrs,
    b_ptrs,
    a_vals,
    b_vals,
    offs_dim0,
    offs_dim1,
    max_dim0,
    max_dim1,
    EVEN_DIM0: tl.constexpr,
    EVEN_DIM1: tl.constexpr,
):
    if EVEN_DIM0:
        if EVEN_DIM1:
            tl.store(a_ptrs, a_vals)
            tl.store(b_ptrs, b_vals)
        else:
            tl.store(a_ptrs, a_vals, mask=offs_dim1[None, :] < max_dim1)
            tl.store(b_ptrs, b_vals, mask=offs_dim1[None, :] < max_dim1)
    else:
        if EVEN_DIM1:
            tl.store(a_ptrs, a_vals, mask=offs_dim0[:, None] < max_dim0)
            tl.store(b_ptrs, b_vals, mask=offs_dim0[:, None] < max_dim0)
        else:
            tl.store(
                a_ptrs,
                a_vals,
                mask=(offs_dim0[:, None] < max_dim0) & (offs_dim1[None, :] < max_dim1),
            )
            tl.store(
                b_ptrs,
                b_vals,
                mask=(offs_dim0[:, None] < max_dim0) & (offs_dim1[None, :] < max_dim1),
            )


@triton.jit
def _bwd_kernel_dtk_and_partial_dk(
    start_n,
    Q,
    K,
    V,
    DTO,
    DK,
    # tangents
    TQ,
    TK,
    # tangents gradients
    DTK,
    # tmp vars
    LSE,
    SUM4,
    softmax_scale,
    # strides
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dkn,
    stride_tqm,
    stride_tkn,
    seqlen_q,
    seqlen_k,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    begin_m = 0
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_qm = offs_m
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dto_ptrs = DTO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])

    tq_ptrs = TQ + (offs_qm[:, None] * stride_tqm + offs_d[None, :])

    dtk_ptrs = DTK + (offs_n[:, None] * stride_tkn + offs_d[None, :])

    acc_dtype = tl.float32
    # acc_dtype = tl.float16
    dtk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=acc_dtype)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=acc_dtype)

    if begin_m >= seqlen_q:
        _bwd_store_2_tensors(
            dk_ptrs,
            dtk_ptrs,
            dk,
            dtk,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_DIM0=EVEN_N,
            EVEN_DIM1=EVEN_HEADDIM,
        )
        return

    if EVEN_N:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, eviction_policy="evict_first")
            v = tl.load(v_ptrs, eviction_policy="evict_first")
        else:
            k = tl.load(
                k_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
    else:
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
                eviction_policy="evict_first",
            )

    intermediate_dtype = tl.float32

    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            tq = tl.load(tq_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                tq = tl.load(tq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                tq = tl.load(
                    tq_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        qk = tl.dot(q, tl.trans(k)).to(intermediate_dtype)  # mathias: a

        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))

        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        p_dtype = p.to(v.dtype)

        if EVEN_M & EVEN_HEADDIM:
            dto = tl.load(dto_ptrs)
        else:
            dto = tl.load(
                dto_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

        dto_mm_v = tl.dot(dto, tl.trans(v)).to(intermediate_dtype)

        sum_4 = tl.load(SUM4 + offs_m_curr).to(intermediate_dtype)
        add_3 = (dto_mm_v - sum_4[:, None]) * p_dtype
        div_7 = add_3 * softmax_scale

        div_7_dtype = div_7.to(q.dtype)
        ktq_grad_k = tl.dot(tl.trans(tq), div_7_dtype)

        dk += tl.trans(ktq_grad_k).to(dk.dtype)

        dtk += tl.dot(tl.trans(div_7_dtype), q).to(dtk.dtype)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        tq_ptrs += BLOCK_M * stride_tqm
        dto_ptrs += BLOCK_M * stride_dom

    # write-back
    _bwd_store_2_tensors(
        dk_ptrs,
        dtk_ptrs,
        dk,
        dtk,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_DIM0=EVEN_N,
        EVEN_DIM1=EVEN_HEADDIM,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "BIAS_TYPE",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _jvp_bwd_kernel_dtk_and_partial_dk(
    Q,
    K,
    V,
    # gradient of tangent out
    DTO,
    # gradients
    DK,
    # tangents
    TQ,
    TK,
    # tangents gradients
    DTK,
    # tmp vars
    LSE,
    SUM4,
    softmax_scale,
    # strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_tqb,
    stride_tqh,
    stride_tqm,
    stride_tkb,
    stride_tkh,
    stride_tkn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DTO += off_b * stride_dob + off_h * stride_doh
    DK += off_b * stride_dkb + off_h * stride_dkh

    # tangents
    TQ += off_b * stride_tqb + off_h * stride_tqh
    TK += off_b * stride_tkb + off_h * stride_tkh
    # tangents gradients
    DTK += off_b * stride_tkb + off_h * stride_tkh

    LSE += off_hb * seqlen_q_rounded

    SUM4 += off_hb * seqlen_q_rounded

    start_n = tl.program_id(0)
    _bwd_kernel_dtk_and_partial_dk(
        start_n,
        Q,
        K,
        V,
        DTO,
        DK,
        # tangents
        TQ,
        TK,
        # tangents gradients
        DTK,
        # tmp vars
        LSE,
        SUM4,
        # strides
        softmax_scale,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dkn,
        # tangents
        stride_tqm,
        stride_tkn,
        seqlen_q,
        seqlen_k,
        headdim,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _bwd_kernel_one_row_dtq_and_partial_dq(
    start_m,
    Q,
    K,
    V,
    DTO,
    DQ,
    # tangents
    TK,
    TV,
    # tangents gradients
    DTQ,
    # tmp vars
    LSE,
    SUM4,
    softmax_scale,
    # strides
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    # tangents
    stride_tqm,
    stride_tkn,
    stride_tvn,
    seqlen_q,
    seqlen_k,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    begin_n = 0
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_qm = offs_m
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dto_ptrs = DTO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])

    tk_ptrs = TK + (offs_n[:, None] * stride_tkn + offs_d[None, :])
    tv_ptrs = TV + (offs_n[:, None] * stride_tvn + offs_d[None, :])

    dtq_ptrs = DTQ + (offs_qm[:, None] * stride_tqm + offs_d[None, :])

    # initialize dv and dk
    acc_dtype = tl.float32
    # acc_dtype = tl.float16
    dq = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=acc_dtype)
    dtq = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=acc_dtype)

    intermediate_dtype = tl.float32

    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    lse_i = tl.load(LSE + offs_m)
    sum_4 = tl.load(SUM4 + offs_m).to(intermediate_dtype)

    if EVEN_M & EVEN_HEADDIM:
        dto = tl.load(dto_ptrs)
    else:
        dto = tl.load(
            dto_ptrs,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        )

    # loop over cols
    num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
    for start_n in range(begin_n, num_block_n * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs, eviction_policy="evict_first")
                v = tl.load(v_ptrs, eviction_policy="evict_first")
                tk = tl.load(tk_ptrs, eviction_policy="evict_first")
            else:
                k = tl.load(
                    k_ptrs,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                v = tl.load(
                    v_ptrs,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tk = tl.load(
                    tk_ptrs,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                    eviction_policy="evict_first",
                )
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs,
                    mask=offs_n_curr[:, None] < seqlen_k,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                v = tl.load(
                    v_ptrs,
                    mask=offs_n_curr[:, None] < seqlen_k,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tk = tl.load(
                    tk_ptrs,
                    mask=offs_n_curr[:, None] < seqlen_k,
                    other=0.0,
                    eviction_policy="evict_first",
                )
            else:
                k = tl.load(
                    k_ptrs,
                    mask=(offs_n_curr[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                    eviction_policy="evict_first",
                )
                v = tl.load(
                    v_ptrs,
                    mask=(offs_n_curr[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tk = tl.load(
                    tk_ptrs,
                    mask=(offs_n_curr[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                    eviction_policy="evict_first",
                )
        qk = tl.dot(q, tl.trans(k)).to(intermediate_dtype)  # mathias: a
        if not (EVEN_N and EVEN_M):  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(
                (offs_n_curr[None, :] < seqlen_k) & (offs_m[:, None] < seqlen_q),
                qk,
                float("-inf"),
            )

        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        p_dtype = p.to(v.dtype)

        dto_mm_v = tl.dot(dto, tl.trans(v)).to(intermediate_dtype)
        add_3 = (dto_mm_v - sum_4[:, None]) * p_dtype
        div_7 = add_3 * softmax_scale

        div_7_dtype = div_7.to(q.dtype)
        qtk_grad_q = tl.dot(div_7_dtype, tk)
        grad_q = qtk_grad_q

        dtq += tl.dot(div_7_dtype.to(k.dtype), k)
        dq += grad_q

        # increment pointers
        k_ptrs += BLOCK_N * stride_kn
        tk_ptrs += BLOCK_N * stride_tkn
        v_ptrs += BLOCK_N * stride_vn
        tv_ptrs += BLOCK_N * stride_tvn

    _bwd_store_2_tensors(
        dq_ptrs,
        dtq_ptrs,
        dq,
        dtq,
        offs_m,
        offs_d,
        seqlen_q,
        headdim,
        EVEN_DIM0=EVEN_M,
        EVEN_DIM1=EVEN_HEADDIM,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "BIAS_TYPE",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _jvp_bwd_kernel_dtq_and_partial_dq(
    Q,
    K,
    V,
    # gradient of tangent out
    DTO,
    # gradients
    DQ,
    # tangents
    TK,
    TV,
    # tangents gradients
    DTQ,
    # tmp vars
    LSE,
    SUM4,
    softmax_scale,
    # strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_tqb,
    stride_tqh,
    stride_tqm,
    stride_tkb,
    stride_tkh,
    stride_tkn,
    stride_tvb,
    stride_tvh,
    stride_tvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DTO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    # pointer to row-wise quantities in value-like data

    # tangents
    TK += off_b * stride_tkb + off_h * stride_tkh
    TV += off_b * stride_tvb + off_h * stride_tvh
    # tangents gradients
    DTQ += off_b * stride_tqb + off_h * stride_tqh

    LSE += off_hb * seqlen_q_rounded

    SUM4 += off_hb * seqlen_q_rounded

    start_m = tl.program_id(0)
    _bwd_kernel_one_row_dtq_and_partial_dq(
        start_m,
        Q,
        K,
        V,
        DTO,
        DQ,
        # tangents
        TK,
        TV,
        # tangents gradients
        DTQ,
        # tmp vars
        LSE,
        SUM4,
        softmax_scale,
        # strides
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dqm,
        # tangents
        stride_tqm,
        stride_tkn,
        stride_tvn,
        seqlen_q,
        seqlen_k,
        headdim,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _bwd_kernel_one_col_dv_and_dtv(
    start_n,
    Q,
    K,
    V,
    DTO,
    DQ,
    DV,
    # tangents
    TQ,
    TK,
    # tangents gradients
    DTQ,
    DTV,
    # tmp vars
    LSE,
    MU,
    softmax_scale,
    # strides
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dvn,
    stride_tqm,
    stride_tkl,
    stride_tvl,
    seqlen_q,
    seqlen_k,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    begin_m = 0
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_qm = offs_m
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dto_ptrs = DTO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])

    tq_ptrs = TQ + (offs_qm[:, None] * stride_tqm + offs_d[None, :])
    tk_ptrs = TK + (offs_n[:, None] * stride_tkl + offs_d[None, :])

    dtq_ptrs = DTQ + (offs_qm[:, None] * stride_tqm + offs_d[None, :])

    # initialize dv and dk
    acc_dtype = tl.float32
    # acc_dtype = tl.float16
    dtv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=acc_dtype)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=acc_dtype)

    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dtv_ptrs = DTV + (offs_n[:, None] * stride_tvl + offs_d[None, :])
        _bwd_store_2_tensors(
            dv_ptrs,
            dtv_ptrs,
            dv,
            dtv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_DIM0=EVEN_N,
            EVEN_DIM1=EVEN_HEADDIM,
        )
        return

    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, eviction_policy="evict_first")
            v = tl.load(v_ptrs, eviction_policy="evict_first")
            tk = tl.load(tk_ptrs, eviction_policy="evict_first")
        else:
            k = tl.load(
                k_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
            tk = tl.load(
                tk_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
    else:
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
            tk = tl.load(
                tk_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
                eviction_policy="evict_first",
            )
            tk = tl.load(
                tk_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
                eviction_policy="evict_first",
            )

    intermediate_dtype = tl.float32

    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            tq = tl.load(tq_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                tq = tl.load(tq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                tq = tl.load(
                    tq_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        qk = tl.dot(q, tl.trans(k)).to(intermediate_dtype)  # mathias: a
        tqk = tl.dot(tq, tl.trans(k)) + tl.dot(q, tl.trans(tk)).to(
            intermediate_dtype
        )  # mathias: c

        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
            tqk = tl.where(offs_n[None, :] < seqlen_k, tqk, 0.0)

        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        mu_ij = tl.load(MU + offs_m_curr).to(intermediate_dtype)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        tqk_scaled = tqk * softmax_scale

        p_dtype = p.to(v.dtype)

        p_tqk = p_dtype * tqk_scaled
        o = p_tqk - p_dtype * mu_ij[:, None]

        if EVEN_M & EVEN_HEADDIM:
            dto = tl.load(dto_ptrs)
        else:
            dto = tl.load(
                dto_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

        dtv += tl.dot(tl.trans(p_dtype), dto).to(dtv.dtype)
        dv += tl.dot(tl.trans(o.to(dto.dtype)), dto).to(dv.dtype)

        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        tq_ptrs += BLOCK_M * stride_tqm
        dto_ptrs += BLOCK_M * stride_dom
        dtq_ptrs += BLOCK_M * stride_tqm

    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dtv_ptrs = DTV + (offs_n[:, None] * stride_tvl + offs_d[None, :])
    _bwd_store_2_tensors(
        dv_ptrs,
        dtv_ptrs,
        dv,
        dtv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_DIM0=EVEN_N,
        EVEN_DIM1=EVEN_HEADDIM,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "BIAS_TYPE",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _jvp_bwd_kernel_dv_and_dtv(
    Q,
    K,
    V,
    # gradient of tangent out
    DTO,
    # gradients
    DQ,
    DV,
    # tangents
    TQ,
    TK,
    # tangents gradients
    DTQ,
    DTV,
    # tmp vars
    LSE,
    MU,
    softmax_scale,
    # strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    # tangents
    stride_tqb,
    stride_tqh,
    stride_tqm,
    stride_tkb,
    stride_tkh,
    stride_tkl,
    stride_tvb,
    stride_tvh,
    stride_tvl,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DTO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DV += off_b * stride_dvb + off_h * stride_dvh
    # pointer to row-wise quantities in value-like data

    # tangents
    TQ += off_b * stride_tqb + off_h * stride_tqh
    TK += off_b * stride_tkb + off_h * stride_tkh
    # tangents gradients
    DTQ += off_b * stride_tqb + off_h * stride_tqh
    DTV += off_b * stride_tvb + off_h * stride_tvh

    LSE += off_hb * seqlen_q_rounded
    MU += off_hb * seqlen_q_rounded

    start_n = tl.program_id(0)
    _bwd_kernel_one_col_dv_and_dtv(
        start_n,
        Q,
        K,
        V,
        DTO,
        DQ,
        DV,
        # tangents
        TQ,
        TK,
        # tangents gradients
        DTQ,
        DTV,
        # tmp vars
        LSE,
        MU,
        softmax_scale,
        # strides
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dqm,
        stride_dvn,
        # tangents
        stride_tqm,
        stride_tkl,
        stride_tvl,
        seqlen_q,
        seqlen_k,
        headdim,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _bwd_kernel_one_col_partial_dk(
    start_n,
    Q,
    K,
    V,
    DTO,
    DK,
    # tangents
    TQ,
    TK,
    TV,
    # tmp vars
    LSE,
    MU,
    LI,
    SUM4,
    SUM7,
    softmax_scale,
    # strides
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dkn,
    # tangents
    stride_tqm,
    stride_tkl,
    stride_tvl,
    seqlen_q,
    seqlen_k,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    begin_m = 0
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_qm = offs_m
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dto_ptrs = DTO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])

    tq_ptrs = TQ + (offs_qm[:, None] * stride_tqm + offs_d[None, :])
    tk_ptrs = TK + (offs_n[:, None] * stride_tkl + offs_d[None, :])
    tv_ptrs = TV + (offs_n[:, None] * stride_tvl + offs_d[None, :])

    dk = tl.load(
        dk_ptrs,
        mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
        other=0.0,
    )

    # So we just exit early.
    if begin_m >= seqlen_q:
        _bwd_store_tensor(
            dk_ptrs,
            dk,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_DIM0=EVEN_N,
            EVEN_DIM1=EVEN_HEADDIM,
        )
        return

    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, eviction_policy="evict_first")
            v = tl.load(v_ptrs, eviction_policy="evict_first")
            tk = tl.load(tk_ptrs, eviction_policy="evict_first")
            tv = tl.load(tv_ptrs, eviction_policy="evict_first")
        else:
            k = tl.load(
                k_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
            tk = tl.load(
                tk_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
            tv = tl.load(
                tv_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
                eviction_policy="evict_first",
            )
    else:
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
            tk = tl.load(
                tk_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
            tv = tl.load(
                tv_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
                eviction_policy="evict_first",
            )
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
                eviction_policy="evict_first",
            )
            tk = tl.load(
                tk_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
                eviction_policy="evict_first",
            )
            tv = tl.load(
                tv_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
                eviction_policy="evict_first",
            )

    intermediate_dtype = tl.float32
    # intermediate_dtype = tl.float16

    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip

        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            tq = tl.load(tq_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                tq = tl.load(tq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                tq = tl.load(
                    tq_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        qk = tl.dot(q, tl.trans(k), out_dtype=intermediate_dtype)  # mathias: a
        tqk = tl.dot(tq, tl.trans(k), out_dtype=intermediate_dtype) + tl.dot(
            q, tl.trans(tk), out_dtype=intermediate_dtype
        )  # mathias: c

        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
            tqk = tl.where(offs_n[None, :] < seqlen_k, tqk, 0.0)

        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        mu_ij = tl.load(MU + offs_m_curr).to(intermediate_dtype)
        l_i = tl.load(LI + offs_m_curr)  # mathias: "k"
        p = tl.exp((qk * softmax_scale - lse_i[:, None]).to(tl.float32))

        tqk_scaled = tqk * softmax_scale

        p_dtype = p.to(v.dtype)

        if EVEN_M & EVEN_HEADDIM:
            dto = tl.load(dto_ptrs)
        else:
            dto = tl.load(
                dto_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

        dto_mm_v = tl.dot(dto, tl.trans(v), out_dtype=intermediate_dtype)

        sum_4 = tl.load(SUM4 + offs_m_curr).to(intermediate_dtype)
        # Note (Mathias): i / l_i == b == p

        l_i_safe = tl.where(offs_m_curr < seqlen_q, l_i, 1.0)

        div_6 = -sum_4 / l_i_safe
        f_t = tqk_scaled.to(intermediate_dtype)
        mul_7 = f_t * div_6[:, None].to(intermediate_dtype)

        m_t_div_k = mu_ij / l_i_safe
        mul_5 = sum_4 * m_t_div_k

        m_i = lse_i - tl.log(l_i_safe)
        add_4 = mul_7 + mul_5[:, None].to(intermediate_dtype)
        mul_8 = add_4 * tl.exp((qk * softmax_scale - m_i[:, None]).to(tl.float32)).to(
            intermediate_dtype
        )

        # TODO @Mathias: technically, we should add the amax backward pass as well, but I will skip that for now
        # CHECK if that has a significant impact on accuracy
        """

            # amax backward
            sum_5 = -mul_8.sum(dim=-1, keepdim=True)
            eq = g == a
            a = None
            sum_6 = eq.sum(dim=-1, keepdim=True)
            div_8 = sum_5 / sum_6
            sum_5 = sum_6 = None
            mul_9 = div_8 * eq
            div_8 = eq = None

            add_5 = mul_8 + mul_9

        """
        add_5 = mul_8

        n = tqk_scaled.to(intermediate_dtype) - mu_ij[:, None]
        n = tl.where((offs_n[None, :] < seqlen_k) & (offs_m_curr[:, None] < seqlen_q), n, 0.0).to(intermediate_dtype)
        dto_mm_v_m_n = dto_mm_v * n
        dto_mm_tv = tl.dot(dto, tl.trans(tv), out_dtype=intermediate_dtype)
        add_2 = dto_mm_tv + dto_mm_v_m_n

        sum_7 = tl.load(SUM7 + offs_m_curr).to(intermediate_dtype)
        fma = (add_2 - sum_7[:, None]) * p_dtype
        add_9 = add_5.to(intermediate_dtype) + fma
        div_9 = add_9 * softmax_scale

        div_9_dtype = div_9.to(q.dtype)

        a1_back_k_grad = tl.trans(
            tl.dot(tl.trans(q), div_9_dtype, out_dtype=intermediate_dtype)
        )
        a1_back_k_grad = tl.where(offs_n[:, None] < seqlen_k, a1_back_k_grad, 0.0)

        dk += a1_back_k_grad.to(dk.dtype)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        tq_ptrs += BLOCK_M * stride_tqm
        dto_ptrs += BLOCK_M * stride_dom

    # write-back
    _bwd_store_tensor(
        dk_ptrs,
        dk,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_DIM0=EVEN_M,
        EVEN_DIM1=EVEN_HEADDIM,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "BIAS_TYPE",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _jvp_bwd_kernel_partial_dk(
    Q,
    K,
    V,
    # gradient of tangent out
    DTO,
    # gradients
    DK,
    # tangents
    TQ,
    TK,
    TV,
    # tmp vars
    LSE,
    MU,
    LI,
    SUM4,
    SUM7,
    softmax_scale,
    # strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    # tangents
    stride_tqb,
    stride_tqh,
    stride_tqm,
    stride_tkb,
    stride_tkh,
    stride_tkl,
    stride_tvb,
    stride_tvh,
    stride_tvl,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DTO += off_b * stride_dob + off_h * stride_doh
    DK += off_b * stride_dkb + off_h * stride_dkh
    # pointer to row-wise quantities in value-like data

    # tangents
    TQ += off_b * stride_tqb + off_h * stride_tqh
    TK += off_b * stride_tkb + off_h * stride_tkh
    TV += off_b * stride_tvb + off_h * stride_tvh

    LSE += off_hb * seqlen_q_rounded
    MU += off_hb * seqlen_q_rounded
    LI += off_hb * seqlen_q_rounded

    SUM4 += off_hb * seqlen_q_rounded
    SUM7 += off_hb * seqlen_q_rounded

    start_n = tl.program_id(0)
    _bwd_kernel_one_col_partial_dk(
        start_n,
        Q,
        K,
        V,
        DTO,
        DK,
        # tangents
        TQ,
        TK,
        TV,
        # tmp vars
        LSE,
        MU,
        LI,
        SUM4,
        SUM7,
        softmax_scale,
        # strides
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dkn,
        # tangents
        stride_tqm,
        stride_tkl,
        stride_tvl,
        seqlen_q,
        seqlen_k,
        headdim,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _bwd_kernel_one_row_partial_dq(
    start_m,
    Q,
    K,
    V,
    DTO,
    DQ,
    # tangents
    TQ,
    TK,
    TV,
    # tmp vars
    LSE,
    MU,
    LI,
    SUM4,
    SUM7,
    softmax_scale,
    # strides
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    # tangents
    stride_tqm,
    stride_tkl,
    stride_tvl,
    seqlen_q,
    seqlen_k,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    begin_n = 0
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_qm = offs_m
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dto_ptrs = DTO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])

    tq_ptrs = TQ + (offs_qm[:, None] * stride_tqm + offs_d[None, :])
    tk_ptrs = TK + (offs_n[:, None] * stride_tkl + offs_d[None, :])
    tv_ptrs = TV + (offs_n[:, None] * stride_tvl + offs_d[None, :])

    # initialize dv and dk
    acc_dtype = tl.float32
    # acc_dtype = tl.float16

    dq = tl.load(
        dq_ptrs,
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(acc_dtype)

    intermediate_dtype = tl.float32

    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
        tq = tl.load(tq_ptrs)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            tq = tl.load(tq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
            tq = tl.load(
                tq_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    lse_i = tl.load(LSE + offs_m)
    mu_ij = tl.load(MU + offs_m).to(intermediate_dtype)
    l_i = tl.load(LI + offs_m)  # mathias: "k"
    sum_4 = tl.load(SUM4 + offs_m).to(intermediate_dtype)
    sum_7 = tl.load(SUM7 + offs_m).to(intermediate_dtype)

    if EVEN_M & EVEN_HEADDIM:
        dto = tl.load(dto_ptrs)
    else:
        dto = tl.load(
            dto_ptrs,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        )

    # loop over cols
    num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
    for start_n in range(begin_n, num_block_n * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs, eviction_policy="evict_first")
                v = tl.load(v_ptrs, eviction_policy="evict_first")
                tk = tl.load(tk_ptrs, eviction_policy="evict_first")
                tv = tl.load(tv_ptrs, eviction_policy="evict_first")
            else:
                k = tl.load(
                    k_ptrs,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                v = tl.load(
                    v_ptrs,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tk = tl.load(
                    tk_ptrs,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tv = tl.load(
                    tv_ptrs,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                    eviction_policy="evict_first",
                )
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs,
                    mask=offs_n_curr[:, None] < seqlen_k,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                v = tl.load(
                    v_ptrs,
                    mask=offs_n_curr[:, None] < seqlen_k,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tk = tl.load(
                    tk_ptrs,
                    mask=offs_n_curr[:, None] < seqlen_k,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tv = tl.load(
                    tv_ptrs,
                    mask=offs_n_curr[:, None] < seqlen_k,
                    other=0.0,
                    eviction_policy="evict_first",
                )
            else:
                k = tl.load(
                    k_ptrs,
                    mask=(offs_n_curr[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                    eviction_policy="evict_first",
                )
                v = tl.load(
                    v_ptrs,
                    mask=(offs_n_curr[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tk = tl.load(
                    tk_ptrs,
                    mask=(offs_n_curr[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tv = tl.load(
                    tv_ptrs,
                    mask=offs_n_curr[:, None] < seqlen_k,
                    other=0.0,
                    eviction_policy="evict_first",
                )
        qk = tl.dot(q, tl.trans(k)).to(intermediate_dtype)  # mathias: a
        tqk = tl.dot(tq, tl.trans(k)) + tl.dot(q, tl.trans(tk)).to(
            intermediate_dtype
        )  # mathias: c

        if not (EVEN_N and EVEN_M):  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(
                (offs_n_curr[None, :] < seqlen_k) & (offs_m[:, None] < seqlen_q),
                qk,
                float("-inf"),
            )
            tqk = tl.where(
                (offs_n_curr[None, :] < seqlen_k) & (offs_m[:, None] < seqlen_q),
                tqk,
                0.0,
            )

        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        tqk_scaled = tqk * softmax_scale

        p_dtype = p.to(v.dtype)

        dto_mm_v = tl.dot(dto, tl.trans(v)).to(intermediate_dtype)

        l_i_safe = tl.where(offs_m < seqlen_q, l_i, 1.0)

        div_6 = -sum_4 / l_i_safe
        f_t = tqk_scaled.to(intermediate_dtype)
        mul_7 = f_t * div_6[:, None].to(intermediate_dtype)

        m_t_div_k = mu_ij / l_i_safe
        mul_5 = sum_4 * m_t_div_k

        m_i = lse_i - tl.log(l_i_safe)
        add_4 = mul_7 + mul_5[:, None]
        mul_8 = add_4 * tl.exp(qk * softmax_scale - m_i[:, None]).to(intermediate_dtype)

        # TODO @Mathias: technically, we should add the amax backward pass as well, but I will skip that for now
        # CHECK if that has a significant impact on accuracy
        """

            # amax backward
            sum_5 = -mul_8.sum(dim=-1, keepdim=True)
            eq = g == a
            a = None
            sum_6 = eq.sum(dim=-1, keepdim=True)
            div_8 = sum_5 / sum_6
            sum_5 = sum_6 = None
            mul_9 = div_8 * eq
            div_8 = eq = None

            add_5 = mul_8 + mul_9

        """
        add_5 = mul_8

        n = tqk_scaled.to(intermediate_dtype) - mu_ij[:, None]
        n = tl.where((offs_n_curr[None, :] < seqlen_k) & (offs_m[:, None] < seqlen_q), n, 0.0).to(intermediate_dtype)
        dto_mm_v_m_n = dto_mm_v * n
        dto_mm_tv = tl.dot(dto, tl.trans(tv)).to(intermediate_dtype)
        add_2 = dto_mm_tv + dto_mm_v_m_n

        fma = (add_2 - sum_7[:, None]) * p_dtype
        add_9 = add_5.to(intermediate_dtype) + fma
        div_9 = add_9 * softmax_scale

        div_9_dtype = div_9.to(q.dtype)

        a1_back_q_grad = tl.dot(div_9_dtype, k)
        a1_back_q_grad = tl.where(offs_m[:, None] < seqlen_q, a1_back_q_grad, 0.0)

        dq += a1_back_q_grad.to(dq.dtype)

        # increment pointers
        k_ptrs += BLOCK_N * stride_kn
        tk_ptrs += BLOCK_N * stride_tkl
        v_ptrs += BLOCK_N * stride_vn
        tv_ptrs += BLOCK_N * stride_tvl

    _bwd_store_tensor(
        dq_ptrs,
        dq,
        offs_m,
        offs_d,
        seqlen_q,
        headdim,
        EVEN_DIM0=EVEN_M,
        EVEN_DIM1=EVEN_HEADDIM,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "BIAS_TYPE",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _jvp_bwd_kernel_row_partial_dq(
    Q,
    K,
    V,
    # gradient of tangent out
    DTO,
    # gradients
    DQ,
    # tangents
    TQ,
    TK,
    TV,
    # tmp vars
    LSE,
    MU,
    LI,
    SUM4,
    SUM7,
    softmax_scale,
    # strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    # tangents
    stride_tqb,
    stride_tqh,
    stride_tqm,
    stride_tkb,
    stride_tkh,
    stride_tkl,
    stride_tvb,
    stride_tvh,
    stride_tvl,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DTO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    # pointer to row-wise quantities in value-like data

    # tangents
    TQ += off_b * stride_tqb + off_h * stride_tqh
    TK += off_b * stride_tkb + off_h * stride_tkh
    TV += off_b * stride_tvb + off_h * stride_tvh

    LSE += off_hb * seqlen_q_rounded
    MU += off_hb * seqlen_q_rounded
    LI += off_hb * seqlen_q_rounded

    SUM4 += off_hb * seqlen_q_rounded
    SUM7 += off_hb * seqlen_q_rounded

    start_m = tl.program_id(0)
    _bwd_kernel_one_row_partial_dq(
        start_m,
        Q,
        K,
        V,
        DTO,
        DQ,
        # tangents
        TQ,
        TK,
        TV,
        # tmp vars
        LSE,
        MU,
        LI,
        SUM4,
        SUM7,
        softmax_scale,
        # strides
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dqm,
        # tangents
        stride_tqm,
        stride_tkl,
        stride_tvl,
        seqlen_q,
        seqlen_k,
        headdim,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "BIAS_TYPE",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _jvp_bwd_preprocess_sum_4_sum_7(
    Q,
    K,
    V,
    # gradient of tangent out
    DTO,
    # tangents
    TQ,
    TK,
    TV,
    # tmp vars
    LSE,
    MU,
    SUM4,
    SUM7,
    softmax_scale,
    # strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    # tangents
    stride_tqb,
    stride_tqh,
    stride_tql,
    stride_tkb,
    stride_tkh,
    stride_tkl,
    stride_tvb,
    stride_tvh,
    stride_tvl,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DTO += off_b * stride_dob + off_h * stride_doh
    # pointer to row-wise quantities in value-like data

    # tangents
    TQ += off_b * stride_tqb + off_h * stride_tqh
    TK += off_b * stride_tkb + off_h * stride_tkh
    TV += off_b * stride_tvb + off_h * stride_tvh

    LSE += off_hb * seqlen_q_rounded
    MU += off_hb * seqlen_q_rounded

    SUM4 += off_hb * seqlen_q_rounded
    SUM7 += off_hb * seqlen_q_rounded

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])

    dto_ptrs = DTO + (offs_m[:, None] * stride_dom + offs_d[None, :])

    tq_ptrs = TQ + (offs_m[:, None] * stride_tql + offs_d[None, :])
    tk_ptrs = TK + (offs_n[:, None] * stride_tkl + offs_d[None, :])
    tv_ptrs = TV + (offs_n[:, None] * stride_tvl + offs_d[None, :])

    lse_ptrs = LSE + offs_m
    mu_ptrs = MU + offs_m

    sum_4_ptrs = SUM4 + offs_m
    sum_7_ptrs = SUM7 + offs_m

    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            tq = tl.load(tq_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            tq = tl.load(tq_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            tq = tl.load(tq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
            tq = tl.load(
                tq_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    lse_i = tl.load(lse_ptrs)
    mu_i = tl.load(mu_ptrs)

    if EVEN_M & EVEN_HEADDIM:
        dto = tl.load(dto_ptrs)
    else:
        dto = tl.load(
            dto_ptrs,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        )

    acc_dtype = tl.float32
    # acc_dtype = tl.float16
    sum_4_acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    sum_7_acc = tl.zeros([BLOCK_M], dtype=acc_dtype)

    # Note: "m" is q_seq_idx, "n" is k_seq_idx
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_N:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
                tk = tl.load(tk_ptrs + start_n * stride_tkl)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                )
                tk = tl.load(
                    tk_ptrs + start_n * stride_tkl,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
                tk = tl.load(
                    tk_ptrs + start_n * stride_tkl,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                tk = tl.load(
                    tk_ptrs + start_n * stride_tkl,
                    mask=((start_n + offs_n)[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        qk = tl.dot(q, tl.trans(k))  # mathias: a

        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where((start_n + offs_n)[None, :] < seqlen_k, qk, float("-inf"))

        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        p = tl.where(offs_m[:, None] < seqlen_q, p, 0.0).to(acc_dtype)
        b = p

        if EVEN_N:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
                tv = tl.load(tv_ptrs + start_n * stride_tvl)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                )
                tv = tl.load(
                    tv_ptrs + start_n * stride_tvl,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
                tv = tl.load(
                    tv_ptrs + start_n * stride_tvl,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                tv = tl.load(
                    tv_ptrs + start_n * stride_tvl,
                    mask=((start_n + offs_n)[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        dto_mm_v = tl.dot(dto, tl.trans(v), out_dtype=acc_dtype)
        dto_mm_v_m_b = dto_mm_v * b

        sum_4 = tl.sum(dto_mm_v_m_b, axis=-1)
        sum_4_acc += sum_4

        tqk = tl.dot(tq, tl.trans(k), out_dtype=acc_dtype) + tl.dot(
            q, tl.trans(tk), out_dtype=acc_dtype
        )  # mathias: c
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            tqk = tl.where((start_n + offs_n)[None, :] < seqlen_k, tqk, float("-inf"))
        n = tl.where(
            ((start_n + offs_n)[None, :] < seqlen_k) & (offs_m[:, None] < seqlen_q), tqk * softmax_scale - mu_i[:, None], 0.0
        ).to(acc_dtype)

        dto_mm_tv = tl.dot(dto, tl.trans(tv), out_dtype=acc_dtype)

        dto_mm_v_m_n = dto_mm_v * n
        add_2 = dto_mm_tv + dto_mm_v_m_n
        mul_11 = add_2 * b
        sum_7 = tl.sum(mul_11, axis=-1)
        sum_7_acc += sum_7

    tl.store(sum_4_ptrs, sum_4_acc)
    tl.store(sum_7_ptrs, sum_7_acc)


def _flash_attn_backward(
    dto: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    tq: torch.Tensor,
    tk: torch.Tensor,
    tv: torch.Tensor,
    lse: torch.Tensor,
    mu: torch.Tensor,
    li: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dtq: torch.Tensor,
    dtk: torch.Tensor,
    dtv: torch.Tensor,
    softmax_scale=None,
):
    # Make sure that the last dimension is contiguous
    if dto.stride(-1) != 1:
        dto = dto.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    if mu.shape[-1] != seqlen_q_rounded:
        assert (
            seqlen_q_rounded % 128 == 0
        ), f"seqlen_q_rounded must be divisible by 128, got {seqlen_q_rounded}"
        mu = torch.cat(
            [
                mu,
                torch.zeros(
                    (batch, nheads, seqlen_q_rounded - mu.shape[-1]),
                    device=mu.device,
                    dtype=mu.dtype,
                ),
            ],
            dim=-1,
        )
    if li.shape[-1] != seqlen_q_rounded:
        li = torch.cat(
            [
                li,
                torch.zeros(
                    (batch, nheads, seqlen_q_rounded - li.shape[-1]),
                    device=li.device,
                    dtype=li.dtype,
                ),
            ],
            dim=-1,
        )

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)

    sum_4 = torch.empty_like(lse)
    sum_7 = torch.empty_like(lse)

    """
    preprocessing kernel
    """

    grid_preprocess = lambda META: (
        triton.cdiv(seqlen_q, META["BLOCK_M"]),
        batch * nheads,
    )
    _jvp_bwd_preprocess_sum_4_sum_7[grid_preprocess](
        q,
        k,
        v,
        dto,
        # tangents
        tq,
        tk,
        tv,
        # tmp vars
        lse,
        mu,
        sum_4,
        sum_7,
        softmax_scale,
        # strides
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        dto.stride(0),
        dto.stride(2),
        dto.stride(1),
        # tangents
        tq.stride(0),
        tq.stride(2),
        tq.stride(1),
        tk.stride(0),
        tk.stride(2),
        tk.stride(1),
        tv.stride(0),
        tv.stride(2),
        tv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        BLOCK_HEADDIM,
    )

    """
    main kernel
    """

    for tan, tan_grad in zip((tq, tk, tv), (dtq, dtk, dtv)):
        for i in range(len(tq.shape)):
            if tan.shape[i] != tan_grad.shape[i]:
                raise ValueError(
                    f"Tangent and tangent gradient have different shapes: {tan.shape} != {tan_grad.shape}"
                )
            if tan.shape[i] > 1 and tan.stride(i) != tan_grad.stride(i):
                raise ValueError(
                    f"Tangent and tangent gradient have different strides: {tan.stride(i)} != {tan_grad.stride(i)}"
                )

    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]),
        batch * nheads,
    )
    _jvp_bwd_kernel_dtk_and_partial_dk[grid](
        q,
        k,
        v,
        dto,
        dk,
        # tangents
        tq,
        tk,
        # tangents gradients
        dtk,
        # tmp vars
        lse,
        sum_4,
        softmax_scale,
        # strides
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        dto.stride(0),
        dto.stride(2),
        dto.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        # tangents
        tq.stride(0),
        tq.stride(2),
        tq.stride(1),
        tk.stride(0),
        tk.stride(2),
        tk.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        BLOCK_HEADDIM,
    )

    grid = lambda META: (
        triton.cdiv(seqlen_q, META["BLOCK_M"]),
        batch * nheads,
    )
    _jvp_bwd_kernel_dtq_and_partial_dq[grid](
        q,
        k,
        v,
        dto,
        dq,
        # tangents
        tk,
        tv,
        # tangents gradients
        dtq,
        # tmp vars
        lse,
        sum_4,
        softmax_scale,
        # strides
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        dto.stride(0),
        dto.stride(2),
        dto.stride(1),
        dq.stride(0),
        dq.stride(2),
        dq.stride(1),
        # tangents
        tq.stride(0),
        tq.stride(2),
        tq.stride(1),
        tk.stride(0),
        tk.stride(2),
        tk.stride(1),
        tv.stride(0),
        tv.stride(2),
        tv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        BLOCK_HEADDIM,
    )

    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]),
        batch * nheads,
    )
    _jvp_bwd_kernel_dv_and_dtv[grid](
        q,
        k,
        v,
        dto,
        dq,
        dv,
        # tangents
        tq,
        tk,
        # tangents gradients
        dtq,
        dtv,
        # tmp vars
        lse,
        mu,
        softmax_scale,
        # strides
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        dto.stride(0),
        dto.stride(2),
        dto.stride(1),
        dq.stride(0),
        dq.stride(2),
        dq.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        # tangents
        tq.stride(0),
        tq.stride(2),
        tq.stride(1),
        tk.stride(0),
        tk.stride(2),
        tk.stride(1),
        tv.stride(0),
        tv.stride(2),
        tv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        BLOCK_HEADDIM,
    )

    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]),
        batch * nheads,
    )
    _jvp_bwd_kernel_partial_dk[grid](
        q,
        k,
        v,
        dto,
        dk,
        # tangents
        tq,
        tk,
        tv,
        # tmp vars
        lse,
        mu,
        li,
        sum_4,
        sum_7,
        softmax_scale,
        # strides
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        dto.stride(0),
        dto.stride(2),
        dto.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        # tangents
        tq.stride(0),
        tq.stride(2),
        tq.stride(1),
        tk.stride(0),
        tk.stride(2),
        tk.stride(1),
        tv.stride(0),
        tv.stride(2),
        tv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        BLOCK_HEADDIM,
    )

    grid = lambda META: (
        triton.cdiv(seqlen_q, META["BLOCK_M"]),
        batch * nheads,
    )
    _jvp_bwd_kernel_row_partial_dq[grid](
        q,
        k,
        v,
        dto,
        dq,
        # tangents
        tq,
        tk,
        tv,
        # tmp vars
        lse,
        mu,
        li,
        sum_4,
        sum_7,
        softmax_scale,
        # strides
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        dto.stride(0),
        dto.stride(2),
        dto.stride(1),
        dq.stride(0),
        dq.stride(2),
        dq.stride(1),
        # tangents
        tq.stride(0),
        tq.stride(2),
        tq.stride(1),
        tk.stride(0),
        tk.stride(2),
        tk.stride(1),
        tv.stride(0),
        tv.stride(2),
        tv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        BLOCK_HEADDIM,
    )
