import torch

from .train import train
from .viz import make_gif, make_gif_3d


def main():
    device = torch.device("cpu")  # later: cuda if available
    model = train(device=device)
    make_gif(model, device=device)
    make_gif_3d(model, device=device)


if __name__ == "__main__":
    main()
