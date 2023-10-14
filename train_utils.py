"""
training utilities
"""

import time
import torch
from torch import nn
from torch.nn import functional as F


def loss_fn(y_hat, y):
    raise NotImplementedError


def accuracy_fn(y_hat, y):
    raise NotImplementedError


def train_step(
    model_G: torch.nn.Module,
    model_D: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module,
    device: torch.device=None,
    real_label: float=1.,
    fake_label: float=0.,
    dim_latent: int=128
):
    """
    training step
    """
    # train mode
    model_G.train()
    model_D.train()

    # loss function
    train_loss_G = 0.
    train_loss_D = 0.

    # iterate over given dataloader
    t0 = time.perf_counter()
    for batch, (X, _) in enumerate(dataloader):
        ########################################
        # update discriminator
        ########################################
        # reset gradients
        optimizer_D.zero_grad()

        # train with all-real batch
        real = X.to(device)
        b_size = real.size(0)
        label = torch.full(size=(b_size,), fill_value=real_label, device=device)
        output = model_D(real).view(-1)
        loss_real = loss_fn(output, label)
        loss_real.backward()
        D_x = output.mean().item()   # average score of discriminator on real data (should be close to 1)

        # train with all-fake batch
        noise = torch.randn(b_size, dim_latent, 1, 1, device=device)
        fake = model_G(noise)
        label.fill_(fake_label)
        output = model_D(fake.detach()).view(-1)
        loss_fake = loss_fn(output, label)

        loss_fake.backward()
        D_G_z1 = output.mean().item()   # average score of discriminator on fake data generated by generator (should be close to 0)

        loss_D = loss_real + loss_fake
        train_loss_D += loss_D.item()
        optimizer_D.step()

        ########################################
        # update generator
        ########################################
        optimizer_G.zero_grad()

        # fake labels are real for generator cost
        label.fill_(real_label)
        output = model_D(fake).view(-1)   # pass fake data to discriminator
        loss_G = loss_fn(output, label)
        loss_G.backward()
        D_G_z2 = output.mean().item()   # average score of discriminator on fake data generated by generator (should be close to 1)
        train_loss_G += loss_G.item()
        optimizer_G.step()

        if batch % 40 == 0:
            t1 = time.perf_counter()
            elps = t1 - t0
            log = f"   >>> batch: {batch}/{len(dataloader)}, " \
                    f"loss_G: {loss_G.item():.4f}, " \
                    f"loss_D: {loss_D.item():.4f}, " \
                    f"D(x): {D_x:.4f}, " \
                    f"D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}, " \
                    f"elapsed: {elps:.2f}"
            print(log)

    train_loss_G /= len(dataloader)
    train_loss_D /= len(dataloader)
    return train_loss_G, train_loss_D


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module,
    device: torch.device=None
):
    raise NotImplementedError


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module,
    epochs: int=5,
    device: torch.device=None,
):
    raise NotImplementedError

