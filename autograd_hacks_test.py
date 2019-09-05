import torch
import torch.nn as nn
import torch.nn.functional as F

import autograd_hacks


# Lenet-5 from https://github.com/pytorch/examples/blob/master/mnist/main.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Tiny LeNet-5 for Hessian testing
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 2, 1)
        self.conv2 = nn.Conv2d(2, 2, 2, 1)
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):            # 28x28
        x = F.max_pool2d(x, 4, 4)    # 7x7
        x = F.relu(self.conv1(x))    # 6x6
        x = F.max_pool2d(x, 2, 2)    # 3x3
        x = F.relu(self.conv2(x))    # 2x2
        x = F.max_pool2d(x, 2, 2)    # 1x1
        x = x.view(-1, 2 * 1 * 1)    # C * W * H
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Autograd helpers, from https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y: torch.Tensor, x: torch.Tensor):
    return jacobian(jacobian(y, x, create_graph=True), x)


def test_grad1():
    torch.manual_seed(1)
    model = Net()
    loss_fn = nn.CrossEntropyLoss()

    n = 4
    data = torch.rand(n, 1, 28, 28)
    targets = torch.LongTensor(n).random_(0, 10)

    autograd_hacks.add_hooks(model)
    output = model(data)
    loss_fn(output, targets).backward(retain_graph=True)
    autograd_hacks.compute_grad1(model)
    autograd_hacks.disable_hooks()

    # Compare values against autograd
    losses = torch.stack([loss_fn(output[i:i+1], targets[i:i+1]) for i in range(len(data))])

    for layer in model.modules():
        if not autograd_hacks.is_supported(layer):
            continue
        for param in layer.parameters():
            assert torch.allclose(param.grad, param.grad1.mean(dim=0))
            assert torch.allclose(jacobian(losses, param), param.grad1)


def test_hess():
    subtest_hess_type('CrossEntropy')
    subtest_hess_type('LeastSquares')


def subtest_hess_type(hess_type):
    torch.manual_seed(1)
    model = TinyNet()

    def least_squares_loss(data_, targets_):
       assert len(data_) == len(targets_)
       err = data_ - targets_
       return torch.sum(err * err) / 2 / len(data_)

    n = 3
    data = torch.rand(n, 1, 28, 28)

    autograd_hacks.add_hooks(model)
    output = model(data)

    if hess_type == 'LeastSquares':
        targets = torch.rand(output.shape)
        loss_fn = least_squares_loss
    else:  # hess_type == 'CrossEntropy':
        targets = torch.LongTensor(n).random_(0, 10)
        loss_fn = nn.CrossEntropyLoss()

    autograd_hacks.backprop_hess(output, hess_type=hess_type)
    autograd_hacks.clear_backprops(model)
    autograd_hacks.backprop_hess(output, hess_type=hess_type)

    autograd_hacks.compute_hess(model)
    autograd_hacks.disable_hooks()

    for layer in model.modules():
        if not autograd_hacks.is_supported(layer):
            continue
        for param in layer.parameters():
            loss = loss_fn(output, targets)
            hess_autograd = hessian(loss, param)
            hess = param.hess
            assert torch.allclose(hess, hess_autograd.reshape(hess.shape))


if __name__ == '__main__':
    test_grad1()
    test_hess()
