"""Safe Query/Key Newton--Schulz update-direction optimizer."""

from __future__ import annotations

import math

import torch


def zeropower_via_newton_schulz(gradient, steps=5, eps=1e-7, variant="polar"):
    if variant != "polar":
        raise ValueError(f"unsupported Newton--Schulz variant: {variant}")
    original_dtype = gradient.dtype
    matrix = gradient.float()
    if matrix.ndim != 2:
        raise ValueError("Safe Q/K expects a two-dimensional projection matrix")
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    matrix = matrix / matrix.norm().clamp_min(float(eps))
    identity = torch.eye(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    for _ in range(int(steps)):
        gram = matrix.T @ matrix
        matrix = 0.5 * matrix @ (3.0 * identity - gram)
    if transposed:
        matrix = matrix.T
    return matrix.to(original_dtype)


class SafeQKWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, groups, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(
            groups,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_safe_qk=False),
        )
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_safe_qk") and self.current_epoch >= int(group.get("start_epoch", 0)):
                self._step_safe_qk(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_safe_qk(self, group):
        lr = float(group["lr"])
        beta = float(group.get("momentum", 0.95))
        for parameter in group["params"]:
            if parameter.grad is None or parameter.grad.ndim != 2:
                continue
            gradient = parameter.grad
            weight_decay = float(group.get("weight_decay", 0.0))
            if weight_decay:
                gradient = gradient.add(parameter, alpha=weight_decay)
            state = self.state[parameter]
            momentum = state.setdefault("momentum", torch.zeros_like(parameter))
            momentum.mul_(beta).add_(gradient)
            update = gradient.add(momentum, alpha=beta) if group.get("nesterov", True) else momentum
            update = zeropower_via_newton_schulz(
                update,
                steps=int(group.get("ns_steps", 5)),
                eps=float(group.get("safe_eps", 1e-7)),
                variant=str(group.get("ns_variant", "polar")),
            )
            scale = math.sqrt(max(1.0, parameter.shape[0] / max(1, parameter.shape[1])))
            parameter.add_(update, alpha=-lr * scale)

    def _step_adamw(self, group):
        lr = float(group.get("adam_lr", group["lr"]))
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("adam_weight_decay", group.get("weight_decay", 0.0)))
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(parameter.grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(parameter.grad, parameter.grad, value=1.0 - beta2)
            correction1 = 1.0 - beta1 ** state["step"]
            correction2 = 1.0 - beta2 ** state["step"]
            denominator = exp_avg_sq.sqrt().div_(math.sqrt(correction2)).add_(eps)
            parameter.addcdiv_(exp_avg, denominator, value=-lr / correction1)


def build_safe_qk_optimizer(model, cfg):
    safe_cfg = cfg["safe_qk"]
    named_qk = list(model.named_qk_parameters())
    qk_ids = {id(parameter) for _, parameter in named_qk}
    auxiliary = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in qk_ids
    ]
    if not named_qk:
        raise RuntimeError("Safe Q/K enabled but no Query/Key matrices were detected")
    betas = tuple(safe_cfg.get("adam_betas", [0.9, 0.999]))
    groups = [
        {
            "name": "safe_qk",
            "params": [parameter for _, parameter in named_qk],
            "lr": float(cfg["lr"]) * float(safe_cfg.get("lr_scale", 3.0)),
            "adam_lr": float(cfg["lr"]),
            "weight_decay": float(safe_cfg.get("weight_decay", 0.0)),
            "adam_weight_decay": float(cfg["weight_decay"]),
            "use_safe_qk": True,
            "momentum": float(safe_cfg.get("momentum", 0.95)),
            "ns_steps": int(safe_cfg.get("ns_steps", 5)),
            "ns_variant": str(safe_cfg.get("ns_variant", "polar")),
            "start_epoch": int(safe_cfg.get("start_epoch", 0)),
            "nesterov": bool(safe_cfg.get("nesterov", True)),
            "safe_eps": float(safe_cfg.get("eps", 1e-7)),
            "betas": betas,
            "eps": float(safe_cfg.get("adam_eps", 1e-8)),
        },
        {
            "name": "adamw_aux",
            "params": [parameter for _, parameter in auxiliary],
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
            "use_safe_qk": False,
            "betas": betas,
            "eps": float(safe_cfg.get("adam_eps", 1e-8)),
        },
    ]
    optimizer = SafeQKWithAuxAdam(
        groups,
        lr=float(cfg["lr"]),
        betas=betas,
        eps=float(safe_cfg.get("adam_eps", 1e-8)),
        weight_decay=float(cfg["weight_decay"]),
    )
    optimizer.safe_qk_parameter_names = [name for name, _ in named_qk]
    optimizer.auxiliary_parameter_names = [name for name, _ in auxiliary]
    return optimizer


def is_qk_parameter(name, parameter):
    lowered = name.lower()
    if parameter.ndim != 2 or "bias" in lowered or "norm" in lowered:
        return False
    return "wq" in lowered or "wk" in lowered or "q_proj" in lowered or "k_proj" in lowered
