"""
Library for extracting interesting quantites from autograd, see README.md

Not thread-safe because of module-level variables

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Hi: per-example Hessian of matmul, shaped as matrix of [dim, dim], indices have been row-vectorized
Hi_bias: per-example Hessian of bias
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Jb: batch output Jacobian of matmul, output sensitivity for example,class pair, [o, n, ....]
Jb_bias: as above, but for bias

A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer

"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Supported layer class types
_supported_layers = ['Linear', 'Conv2d']
# work-around for https://github.com/pytorch/pytorch/issues/25723
_hooks_disabled: bool = False
# global switch to catch double backprop errors on Hessian computation
_enforce_fresh_backprop: bool = False
_enforce_fresh_activation: bool = False


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.

    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.

    Call "remove_hooks(model)" to disable this.

    Args:
        model:
    """

    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_backward_hook(_capture_backprops))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)


def remove_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks(model)
    """

    assert model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"

    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle.remove()
        del model.autograd_hacks_hooks


def disable_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """

    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks() -> None:
    """the opposite of disable_hooks()"""

    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""
    global _enforce_fresh_activation
    if _hooks_disabled:
        return

    if _enforce_fresh_activation:
        assert not hasattr(layer, 'activations_list'), """
        previous forward pass detected"""
        _enforce_fresh_activation = False

    if not hasattr(layer, 'activations_list'):
        layer.activations_list = []

    assert _layer_type(layer) in _supported_layers, """
    Hook installed on unsupported layer, this shouldn't happen"""
    layer.activations_list.append(input[0].detach())


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _enforce_fresh_backprop
    if _hooks_disabled:
        return

    if _enforce_fresh_backprop:
        assert not hasattr(layer, 'backprops_list'), "Seeing result of previous backprop, use clear_backprops(model) to clear"
        _enforce_fresh_backprop = False

    if not hasattr(layer, 'backprops_list'):
        layer.backprops_list = []

    layer.backprops_list.append(output[0].detach())


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list


def compute_grad1_for_linear(layer, A, B):
    weight_grad = torch.einsum('ni,nj->nij', B, A)
    append(layer.weight, 'grad1', weight_grad)
    if layer.bias is not None:
        append(layer.bias, 'grad1', B)


def compute_grad1_for_conv2d(layer, A, B):
    n = A.shape[0]
    A = torch.nn.functional.unfold(A, layer.kernel_size)
    B = B.reshape(n, -1, A.shape[-1])
    grad1 = torch.einsum('ijk,ilk->ijl', B, A)
    shape = [n] + list(layer.weight.shape)
    append(layer.weight, 'grad1', grad1.reshape(shape))
    if layer.bias is not None:
        append(layer.bias, 'grad1', torch.sum(B, dim=2))


def clear_grad1_for_linear_and_conv2d(layer):
    if hasattr(layer.weight, 'grad1'):
        del layer.weight.grad1
    if layer.bias is not None and hasattr(layer.bias, 'grad1'):
        del layer.bias.grad1


def append(instance, attrib, item):
    if not hasattr(instance, attrib):
        setattr(instance, attrib, [])

    getattr(instance, attrib).append(item)


compute_grad1_for = {
    'Linear': compute_grad1_for_linear,
    'Conv2d': compute_grad1_for_conv2d
}

clear_grad_for = {
    'Linear': clear_grad1_for_linear_and_conv2d,
    'Conv2d': clear_grad1_for_linear_and_conv2d
}


def compute_grad1(model: nn.Module, loss_type: str = 'mean') -> None:
    """Compute per-example gradients and save them under 'param.grad1'. Must be
    called after loss.backprop()

    Args:
        model:
        loss_type: either "mean" or "sum" depending on the backpropped loss
    """
    assert loss_type in ('mean', 'sum')
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations_list'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        assert len(layer.activations_list) == len(layer.backprops_list)

        clear_grad_for[layer_type](layer)
        for A, B in zip(layer.activations_list, layer.backprops_list):
            if loss_type == 'mean':
                B *= A.shape[0]

            compute_grad1_for[layer_type](layer, A, B)


def compute_hess(model: nn.Module,) -> None:
    """Save Hessian under param.hess for each param in the model"""

    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations_list'), "No forward passes detected"
        assert len(layer.activations_list) == 1
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"

        if layer_type == 'Linear':
            A = layer.activations_list[0]
            B = torch.stack(layer.backprops_list)

            n = A.shape[0]
            o = B.shape[0]

            A = torch.stack([A] * o)
            Jb = torch.einsum("oni,onj->onij", B, A).reshape(n*o,  -1)
            layer.weight.hess = torch.einsum('ni,nj->ij', Jb, Jb) / n
            if layer.bias is not None:
                setattr(layer.bias, 'hess', torch.einsum('oni,onj->ij', B, B)/n)

        elif layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels

            A = layer.activations_list[0].detach()
            A = torch.nn.functional.unfold(A, (Kh, Kw))       # n, di * Kh * Kw, Oh * Ow
            n = A.shape[0]
            B = torch.stack([Bt.reshape(n, do, -1) for Bt in layer.backprops_list])  # o, n, do, Oh*Ow
            o = B.shape[0]

            A = torch.stack([A] * o)                          # o, n, di * Kh * Kw, Oh*Ow
            Jb = torch.einsum('onij,onkj->onik', B, A)        # o, n, do, di * Kh * Kw

            Hi = torch.einsum('onij,onkl->nijkl', Jb, Jb)     # n, do, di*Kh*Kw, do, di*Kh*Kw
            Jb_bias = torch.einsum('onij->oni', B)
            Hi_bias = torch.einsum('oni,onj->nij', Jb_bias, Jb_bias)

            layer.weight.hess = Hi.mean(dim=0)
            if layer.bias is not None:
                layer.bias.hess = Hi_bias.mean(dim=0)


def backprop_hess(output: torch.Tensor, hess_type: str) -> None:
    """
    Call backprop 1 or more times to get values needed for Hessian computation.

    Args:
        output: prediction of neural network (ie, input of nn.CrossEntropyLoss())
        hess_type: type of Hessian propagation, "CrossEntropy" results in exact Hessian for CrossEntropy

    Returns:

    """

    assert hess_type in ('LeastSquares', 'CrossEntropy')
    global _enforce_fresh_backprop
    n, o = output.shape

    _enforce_fresh_backprop = True

    if hess_type == 'CrossEntropy':
        batch = F.softmax(output, dim=1)

        mask = torch.eye(o).expand(n, o, o)
        diag_part = batch.unsqueeze(2).expand(n, o, o) * mask
        outer_prod_part = torch.einsum('ij,ik->ijk', batch, batch)
        hess = diag_part - outer_prod_part
        assert hess.shape == (n, o, o)
        for i in range(n):
            hess[i, :, :] = symsqrt(hess[i, :, :])

        hess = hess.transpose(0, 1)

    elif hess_type == 'LeastSquares':
        hess = []
        assert len(output.shape) == 2
        batch_size, output_size = output.shape
        id_mat = torch.eye(output_size)
        for out_idx in range(output_size):
            hess.append(torch.stack([id_mat[out_idx]] * batch_size))

    for o in range(o):
        output.backward(hess[o], retain_graph=True)


def symsqrt(a, cond=None, return_rank=False, dtype=torch.float32):
    """Symmetric square root of a positive semi-definite matrix.
    See https://github.com/pytorch/pytorch/issues/25481"""

    s, u = torch.symeig(a, eigenvectors=True)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

    if cond in [None, -1]:
        cond = cond_dict[dtype]

    above_cutoff = (abs(s) > cond * torch.max(abs(s)))

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]

    B = u @ torch.diag(psigma_diag) @ u.t()
    return (B, len(psigma_diag)) if return_rank else B
