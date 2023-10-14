"""
DCGAN: Deep Convolutional Generative Adversarial Network
"""

import os
import pathlib
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchinfo import summary

import models
import train_utils
from models import *
from train_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-p", "--patience", type=int, default=10)
parser.add_argument("-r", "--lr", type=float, default=2e-4)
parser.add_argument("-d", "--device", type=str, default="mps")
parser.add_argument("-s", "--seed", type=int, default=42)

parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--dim_latent", type=int, default=128)
parser.add_argument("--dim_gen", type=int, default=64)
parser.add_argument("--dim_dis", type=int, default=64)
parser.add_argument("--fraction", type=float, default=1.)


def main():
    # plot settings
    plot_settings()

    # main
    _main()


def _main():
    # arguments
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set device
    device = args.device
    print(f">>> using device: {device}")

    # dataroot
    path_images = pathlib.Path("images")
    path_images = path_images / "data"
    print(f">>> path_images: {path_images}")

    # dataset
    image_size = args.image_size
    print(f">>> image_size: {image_size}")
    dataset = datasets.ImageFolder(
        root=path_images,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    )
    print(f">>> dataset: {len(dataset)}")

    # smaller dataset
    frac = args.fraction
    print(f">>> fraction: {frac}")
    dataset = torch.utils.data.Subset(
        dataset,
        np.random.choice(
            len(dataset), size=int(frac * len(dataset)), replace=False
        )
    )
    print(f">>> new dataset: {len(dataset)}")

    # dataloader
    print(f">>> batch_size: {args.batch_size}")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    print(f">>> dataloader: {len(dataloader)}")

    # plot some images
    path_results = pathlib.Path("results")
    path_results.mkdir(exist_ok=True)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.imshow(
        np.transpose(
            make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0)
        )
    )
    plt.axis("off")
    plt.title("Training Images")
    plt.tight_layout()
    plt.savefig(path_results / "training_images.png")
    plt.close()

    # model
    netG = Generator(
        dim_channel=args.channels,
        dim_latent=args.dim_latent,
        dim_feature=args.dim_gen
    ).to(device)
    print(netG)

    netD = Discriminator(
        dim_channel=args.channels,
        dim_feature=args.dim_dis
    ).to(device)
    print(netD)

    # apply the weights_init function to randomly initialize all weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # loss function
    loss_fn = nn.BCELoss()

    # noise
    noise = torch.randn(
        args.batch_size,
        args.dim_latent,
        1,
        1,
        device=device
    )

    # labels
    real_label = 1.
    fake_label = 0.

    # optimizers
    optimizerG = torch.optim.Adam(
        params=netG.parameters(),
        lr=args.lr,
        betas=(.5, .999)
    )
    optimizerD = torch.optim.Adam(
        params=netD.parameters(),
        lr=args.lr,
        betas=(.5, .999)
    )

    # checkpoint
    path_checkpoints = pathlib.Path("checkpoints")
    path_checkpoints.mkdir(exist_ok=True)

    # training loop
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    wait = 0
    best_loss = np.inf   # we focus on generator's performance
    for epoch in range(0, args.epochs+1):
        print(f"\n>>> epoch: {epoch}/{args.epochs}")
        loss_D, loss_G = train_step(
            model_G=netG,
            model_D=netD,
            dataloader=dataloader,
            optimizer_G=optimizerG,
            optimizer_D=optimizerD,
            loss_fn=loss_fn,
            accuracy_fn=None,
            device=device,
            real_label=real_label,
            fake_label=fake_label,
            dim_latent=args.dim_latent
        )

        if loss_G < best_loss:
            best_loss = loss_G
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(">>> early stopping")
                break

        # generate images
        with torch.inference_mode():
            fake = netG(noise).detach().cpu()
            # save_image(
            #     fake,
            #     path_results / f"fake_samples_epoch_{epoch}.png",
            #     normalize=True
            # )

            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(f"Generated Images at epoch {epoch}")
            plt.imshow(
                np.transpose(
                    make_grid(fake[:64], padding=2, normalize=True).cpu(),
                    (1, 2, 0)
                )
            )
            plt.tight_layout()
            plt.savefig(path_results / f"generated_images_epoch_{epoch}.png")
            plt.close()

        # save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_G": netG.state_dict(),
                "model_D": netD.state_dict(),
                "optimizer_G": optimizerG.state_dict(),
                "optimizer_D": optimizerD.state_dict(),
                "loss_G": loss_G,
                "loss_D": loss_D
            },
            path_checkpoints / f"checkpoint_epoch_{epoch}.pt"
        )

        # save losses
        G_losses.append(loss_G)
        D_losses.append(loss_D)

    # plot losses
    plt.figure()
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path_results / "loss.png")
    plt.close()

    # get a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.axis("off")
    plt.title("Real Images")

    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(make_grid(fake[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.axis("off")
    plt.title("Fake Images")

    plt.tight_layout()
    plt.savefig(path_results / "comparison.png")
    plt.close()

    # save the model
    path_models = pathlib.Path("models")
    path_models.mkdir(exist_ok=True)
    torch.save(netG.state_dict(), path_models / "trained_generator.pt")
    torch.save(netD.state_dict(), path_models / "trained_discriminator.pt")


def plot_settings():
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["savefig.dpi"] = 300


if __name__ == "__main__":
    main()
