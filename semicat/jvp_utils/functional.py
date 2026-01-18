import math
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor


class TritonJVPSDPAInnerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        t_q: torch.Tensor,
        t_k: torch.Tensor,
        t_v: torch.Tensor,
        o: torch.Tensor,
        M: torch.Tensor,
        MU: torch.Tensor,
        LI: torch.Tensor,
    ):
        from semicat.jvp_utils.ryu_triton import (
            flash_attention_jvp_multihead_triton_kernel_wrapper,
        )

        with torch.no_grad():
            o_new, t_o, M, MU, LI = flash_attention_jvp_multihead_triton_kernel_wrapper(
                Q=q,
                K=k,
                V=v,
                t_Q=t_q,
                t_K=t_k,
                t_V=t_v,
                y=o,
                M=M,
                MU=MU,
                LI=LI,
                return_M=True,
            )
        return t_o, M, MU, LI

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pass


class SDPAJVPForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v, t_q, t_k, t_v, y, M):
        B, H, S, D = q.shape
        S_kv = k.shape[-2]
        S_div_up = ((S + 127) // 128) * 128
        MU = torch.zeros((B, H, S_div_up), device=q.device, dtype=torch.float32)
        LI = torch.full((B, H, S_div_up), fill_value=S_kv, device=q.device, dtype=torch.float32)
        t_y, M, MU, LI = TritonJVPSDPAInnerFunction.apply(
            q, k, v, t_q, t_k, t_v, y, M, MU, LI
        ) 

        return t_y, M, MU, LI

    @staticmethod
    def setup_context(ctx, inputs: tuple[Any, ...], output: Any) -> Any:
        t_y, M, MU, LI = output
        ctx.save_for_backward(*inputs, t_y, MU, LI)

    @staticmethod
    def backward(ctx, d_t_y, *args):
        q, k, v, t_q, t_k, t_v, y, M, t_y, MU, LI = ctx.saved_tensors
        grads = SDPAJVPFlashBackwardFunction.apply(
            q,
            k,
            v,
            t_q,
            t_k,
            t_v,
            d_t_y,
            M,
            MU,
            LI,
            t_y,
        )
        return *grads, None, None, None, None, None


class SDPAJVPFlashBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        t_q: torch.Tensor,
        t_k: torch.Tensor,
        t_v: torch.Tensor,
        tangents_t_o: torch.Tensor,
        lse: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        li: torch.Tensor | None = None,
        o: torch.Tensor | None = None,
    ):
        q_flash = q.transpose(-2, -3)
        k_flash = k.transpose(-2, -3)
        v_flash = v.transpose(-2, -3)
        o_flash = o.transpose(-2, -3)
        tangents_t_o_flash = tangents_t_o.transpose(-2, -3)
        t_q_flash = t_q.transpose(-2, -3)
        t_k_flash = t_k.transpose(-2, -3)
        t_v_flash = t_v.transpose(-2, -3)

        d_q_flash = torch.empty_like(q_flash)
        d_k_flash = torch.empty_like(k_flash)
        d_v_flash = torch.empty_like(v_flash)

        d_t_q_flash = torch.empty_like(t_q_flash)
        d_t_k_flash = torch.empty_like(t_k_flash)
        d_t_v_flash = torch.empty_like(t_v_flash)

        lse_flash = lse
        S_lse = lse_flash.shape[-1]
        assert S_lse % 128 == 0, f"lse seq length must be divisible by 128, got {S_lse}"

        from semicat.jvp_utils.flash_jvp_backward import (
            _flash_attn_backward as flash_jvp_backward,
        )

        flash_jvp_backward(
            tangents_t_o_flash,
            q_flash,
            k_flash,
            v_flash,
            o_flash,
            t_q_flash,
            t_k_flash,
            t_v_flash,
            lse_flash,
            mu,
            li,
            d_q_flash,
            d_k_flash,
            d_v_flash,
            d_t_q_flash,
            d_t_k_flash,
            d_t_v_flash,
        )
        d_v_flash = d_v_flash.transpose(-2, -3)
        d_k_flash = d_k_flash.transpose(-2, -3)
        d_q_flash = d_q_flash.transpose(-2, -3)
        d_t_v_flash = d_t_v_flash.transpose(-2, -3)
        d_t_k_flash = d_t_k_flash.transpose(-2, -3)
        d_t_q_flash = d_t_q_flash.transpose(-2, -3)

        return (
            d_q_flash,
            d_k_flash,
            d_v_flash,
            d_t_q_flash,
            d_t_k_flash,
            d_t_v_flash,
        )

    @staticmethod
    def backward(ctx, d_q, d_k, d_v, d_t_q, d_t_k, d_t_v):
        raise NotImplementedError("Not implemented")


class SDPABackwardFunction(torch.autograd.Function):
    def forward(ctx, q, k, v, o, do, M):
        with torch.inference_mode():
            q = q.transpose(-2, -3)
            k = k.transpose(-2, -3)
            v = v.transpose(-2, -3)
            o = o.transpose(-2, -3)
            do = do.transpose(-2, -3)
            lse = M
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            from semicat.jvp_utils.flash_attn_triton import (
                _flash_attn_backward,
            )

            S_lse = lse.shape[-1]
            assert (
                S_lse % 128 == 0
            ), f"lse seq length must be divisible by 128, got {S_lse}"

            _flash_attn_backward(
                do,
                q,
                k,
                v,
                o,
                lse,
                dq,
                dk,
                dv,
                bias=None,
                causal=False,
                softmax_scale=None,
            )
            dq = dq.transpose(-2, -3)
            dk = dk.transpose(-2, -3)
            dv = dv.transpose(-2, -3)  
        return dq, dk, dv

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError("Not implemented")


class SDPAFunction(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v):
        """
        Creates some empty tensors which will be used only in the jvp forward.
        We have to allocate all of them here instead of in the setup_context function,
        because the backward of this function would otherwise not a different copy of the tensors with invalid values.
        """
        B, H, S, D = q.shape
        S_kv = k.shape[-2]
        D_v = v.shape[-1]
        y = torch.empty((B, H, S, D_v), device=q.device, dtype=q.dtype)
        S_div_up = ((S + 127) // 128) * 128 
        M = torch.full((B, H, S_div_up), fill_value=math.log(S_kv), device=q.device, dtype=torch.float32)
 
        return y, M

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        ctx.save_for_forward(*inputs, *output)
        ctx.save_for_backward(*inputs, *output)
        ctx.set_materialize_grads(False)

    @staticmethod
    def jvp(ctx, t_q, t_k, t_v):
        (
            q,
            k,
            v,
            y,
            M,
        ) = ctx.saved_for_forward 
        t_y, _, _, _ = SDPAJVPForwardFunction.apply(  # type: ignore
            q,
            k,
            v,
            t_q,
            t_k,
            t_v,
            y,
            M,
        )   
        return t_y, M

    @staticmethod
    def backward(ctx, d_y, d_M):
        with torch.inference_mode():
            if d_y is None:
                return None, None, None
            q, k, v, y, M = ctx.saved_tensors
            dq, dk, dv = SDPABackwardFunction.apply(q, k, v, y, d_y, M)  # type: ignore 

        return dq, dk, dv

 

@torch._dynamo.disable
def sdpa_jvp(
    q: Float[Tensor, "B H S D"],
    k: Float[Tensor, "B H S D"],
    v: Float[Tensor, "B H S D"],
) -> Float[Tensor, "B H S D"]:
    """
    Fused scaled dot product attention with jvp support.

    Pytorch's torch.nn.functional.scaled_dot_product_attention does not support jvp which is why we had to implement it ourselves.
    Please note that jvp is VERY heavy even with special triton kernels. The main benefit of these kernels over vanilla sdpa is that the peak memory usage is much lower.

    Supports FP16 and BF16. Not tested with FP32.

    WARNING:
    - running this kernel without JVP will produce wrong results!

    How this sdpa works:
    - First, the normal "forward" of the sdpa is called. After that, the autograd function tools call "setup_context" followed by "jvp"
    - As we can fuse the triton kernels for "normal" and "jvp" forward, we pre-allocate the outputs of the
      normal forward in the "forward", but initialize them only later in the "jvp" method using our fused kernel.
    - In the backward pass, jvp and normal are processed separately since pytorch does not provide a way to track and combine the gradients of the two outputs.
    - The forward of the jvp sdpa uses a customized version of Ryu's jvp triton kernel (ryu_triton.py).
    - The backward of the normal sdpa uses Tri Dao's triton flash attention kernel (flash_attn_triton.py).
    - The backward of the jvp sdpa uses our own triton kernel (flash_jvp_backward.py).

    usage example:
    ```
    y, t_y = torch.func.jvp(sdpa_jvp, (q, k, v), (t_q, t_k, t_v))
    ```
    Or just wrap the entire model that is using this sdpa with torch.func.jvp

    """

    y, _ = SDPAFunction.apply(q, k, v)
    return y


def safe_sdpa_jvp(
    q: Float[Tensor, "B S H D"],
    k: Float[Tensor, "B S H D"],
    v: Float[Tensor, "B S H D"],
) -> Float[Tensor, "B S H D"]:
    """
    A wrapper around `sdpa_jvp` that casts inputs to bfloat16 if they are not yet.
    """
    orig_dtype = q.dtype
    if orig_dtype != torch.bfloat16 and orig_dtype != torch.float16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    q = q.transpose(1, 2)  # B S H D -> B H S D
    k = k.transpose(1, 2)  # B S H D -> B H S D
    v = v.transpose(1, 2)  # B S H D -> B H S D

    y = sdpa_jvp(q, k, v)

    if y.dtype != orig_dtype:
        y = y.to(orig_dtype)
    return y.transpose(1, 2)  # B H S D -> B S H D
