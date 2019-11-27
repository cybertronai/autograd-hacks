# autograd-hacks

Extract useful quantities from PyTorch autograd

## Per-example gradients

```
autograd_hacks.add_hooks(model)
output = model(data)
loss_fn(output, targets).backward()
autograd_hacks.compute_grad1()

# param.grad: gradient averaged over the batch
# param.grad1[i]: gradient with respect to example i

for param in model.parameters():
  assert(torch.allclose(param.grad1.mean(dim=0), param.grad))
```


## Hessians
(assuming ReLU activations, oherwise produces Gauss-Newton matrix)

```
autograd_hacks.backprop_hess(model(data), hess_type='CrossEntropy')
autograd_hacks.compute_hess(model)
print(param.hess)  # print Hessian of param
```
