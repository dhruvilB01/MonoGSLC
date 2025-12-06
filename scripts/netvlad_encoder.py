import warnings
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


class NetVLAD(nn.Module):
    """
    Lightweight NetVLAD pooling layer.
    Adapted from common open-source implementations; centers are learnable.
    """

    def __init__(self, num_clusters: int = 64, dim: int = 512, alpha: float = 100.0, normalize_input: bool = True):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input

        # soft-assignment
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts: torch.Tensor, traindescs: torch.Tensor) -> None:
        """
        Optional helper to initialize centroids and conv from precomputed clusters.
        clsts: [K, D], traindescs: [N, D]
        """
        if clsts.shape != self.centroids.shape:
            raise ValueError(f"Expected clsts shape {self.centroids.shape}, got {clsts.shape}")
        self.centroids = nn.Parameter(clsts)
        # Initialize conv weights so that softmax assignment focuses on closest centroids
        w = 2.0 * self.alpha * clsts
        b = -self.alpha * (clsts**2).sum(dim=1)
        self.conv.weight = nn.Parameter(w.view(self.num_clusters, self.dim, 1, 1))
        self.conv.bias = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # soft-assignment
        soft = self.conv(x).view(N, self.num_clusters, -1)
        soft = F.softmax(soft, dim=1)

        x_flatten = x.view(N, C, -1)  # [N, C, HW]
        vlad = torch.zeros([N, self.num_clusters, C], device=x.device, dtype=x.dtype)
        for k in range(self.num_clusters):
            residual = x_flatten - self.centroids[k : k + 1, :].unsqueeze(-1)
            residual *= soft[:, k : k + 1, :]
            vlad[:, k, :] = residual.sum(dim=-1)

        # intra-normalization then L2
        vlad = F.normalize(vlad, p=2, dim=2)  # [N, K, C]
        vlad = vlad.view(N, -1)  # [N, K*C]
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


def _strip_fc_backbone(backbone: nn.Module) -> Tuple[nn.Module, int]:
    """
    Remove classifier head; return feature extractor and output channels.
    Works with torchvision ResNet-family backbones.
    """
    if isinstance(backbone, models.ResNet):
        layers = [
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ]
        feat = nn.Sequential(*layers)
        out_ch = backbone.layer4[-1].conv3.out_channels if hasattr(backbone.layer4[-1], "conv3") else backbone.layer4[-1].conv2.out_channels
        return feat, out_ch
    raise ValueError(f"Backbone type {type(backbone)} not supported")


def get_netvlad_model(
    num_clusters: int = 64,
    backbone_name: str = "resnet18",
    pretrained_backbone: bool = True,
    normalize_input: bool = True,
    device: Optional[str] = None,
) -> Tuple[nn.Module, transforms.Compose, str, int]:
    """
    Build a NetVLAD encoder (backbone + NetVLAD pooling).
    Returns: (model, transform, device, embed_dim)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Backbone
    weights = None
    if pretrained_backbone:
        try:
            if backbone_name == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            elif backbone_name == "resnet34":
                weights = models.ResNet34_Weights.IMAGENET1K_V1
            elif backbone_name == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V2
        except Exception:
            warnings.warn("Could not load pretrained weights; using random init for backbone.")
            weights = None
    if backbone_name == "resnet18":
        backbone = models.resnet18(weights=weights)
    elif backbone_name == "resnet34":
        backbone = models.resnet34(weights=weights)
    elif backbone_name == "resnet50":
        backbone = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    feat_extractor, feat_dim = _strip_fc_backbone(backbone)
    vlad = NetVLAD(num_clusters=num_clusters, dim=feat_dim, normalize_input=normalize_input)

    class NetVLADModel(nn.Module):
        def __init__(self, feat: nn.Module, vlad_layer: NetVLAD):
            super().__init__()
            self.feat = feat
            self.vlad = vlad_layer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.feat(x)
            return self.vlad(x)

    model = NetVLADModel(feat_extractor, vlad)
    model = model.to(device).eval()

    # Transforms: resize shorter side to 480, normalize ImageNet
    tfm = transforms.Compose(
        [
            transforms.Resize(480),
            transforms.CenterCrop((480, 640)) if backbone_name in ("resnet18", "resnet34", "resnet50") else transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    embed_dim = num_clusters * feat_dim
    return model, tfm, device, embed_dim


@torch.no_grad()
def encode_image(
    img: Union[str, Image.Image],
    model: nn.Module,
    transform,
    device: str,
) -> torch.Tensor:
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    else:
        img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)  # [1, D]
    return feat.squeeze(0).cpu()
