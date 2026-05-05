import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _make_empty_like_linear(linear):
    new_linear = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    new_linear = new_linear.to(linear.weight.dtype).to(linear.weight.device)
    new_linear.weight.data.copy_(linear.weight.data)
    if linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.data)
    return new_linear


def _svd_lowrank_with_retries(w, rank):
    max_rank = min(w.size())
    rank = max(1, min(rank, max_rank))
    candidates = []
    for q in [rank, int(rank * 0.9), int(rank * 0.75), int(rank * 0.5)]:
        q = max(1, min(q, max_rank))
        if q not in candidates:
            candidates.append(q)

    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    last_error = None
    for q in candidates:
        try:
            return torch.svd_lowrank(w, q=q)
        except RuntimeError as exc:
            last_error = exc
            if w.is_cuda:
                torch.cuda.empty_cache()

    try:
        U, S, Vh = torch.linalg.svd(w.cpu(), full_matrices=False)
        q = candidates[-1]
        return U[:, :q].to(w.device), S[:q].to(w.device), Vh[:q, :].t().to(w.device)
    except RuntimeError as exc:
        last_error = exc

    raise last_error


class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV") -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
        self.truncation_rank = S.size(0)
        if sigma_fuse == "UV":
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)

        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division
            w = w * scaling_diag_matrix.view(1, -1)
        Us = []
        Ss = []
        Vs = []
        try:
            U, S, V = _svd_lowrank_with_retries(w, rank)
        except RuntimeError as exc:
            print(f"svd failed for {linear}, keep original linear. error={exc}")
            return _make_empty_like_linear(linear)
        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if not torch.isfinite(S).all():
                print("nan in S")
                return _make_empty_like_linear(linear)
        for U in Us:
            if not torch.isfinite(U).all():
                print("nan in U")
                return _make_empty_like_linear(linear)
        for V in Vs:
            if not torch.isfinite(V).all():
                print("nan in V")
                return _make_empty_like_linear(linear)

        assert len(Us) == len(Ss) == len(Vs) == 1
        new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        return new_linear

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y


class GradSVDLinear(nn.Module):
    def __init__(self, weight, scale, bias, rank) -> None:
        super().__init__()
        self.weight = weight
        self.scale = nn.Parameter(scale)
        self.bias = bias
        self.rank = rank

    @staticmethod
    def from_linear(
        linear: nn.Linear, param_ratio: float, act_aware=False, ic_split=1, oc_split=1, alpha=1, sigma_fuse="UV"
    ):
        if param_ratio >= 1:
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        return GradSVDLinear(w, scaling_diag_matrix, bias, rank)

    def forward(self, inp):
        w = self.weight * self.scale.view(1, -1)
        U, S, V = torch.svd_lowrank(w, q=self.rank)
        new_w = U.mul(S).mm(V.t())
        y = F.linear(inp, new_w, self.bias)
        return y
