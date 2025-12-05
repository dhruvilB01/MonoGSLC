import torch
import numpy as np
from typing import Union, Tuple, List
from PIL import Image

try:
    import timm
    from timm.data import resolve_model_data_config
    from timm.data.transforms_factory import create_transform
except ImportError as e:
    raise ImportError(
        "timm is required. Install with: pip install timm"
    ) from e

# Map short names to timm model IDs
MODEL_MAP = {
    "dinov2": "vit_base_patch14_dinov2",           # 768-d embeddings
    "dinov2-b14": "vit_base_patch14_dinov2",
    "clip-b32": "vit_base_patch32_clip_224.openai", # ~512-d embeddings
}


def _resolve_model_name(model_name: str) -> str:
    # Allow both short names and explicit timm IDs
    return MODEL_MAP.get(model_name, model_name)


def get_model_and_transform(
    model_name: str = "dinov2", device: Union[str, None] = None
):
    """
    Load a pre-trained vision encoder and its preprocessing transform.

    Returns: (model, transform, device, embed_dim)
    """
    model_id = _resolve_model_name(model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # num_classes=0 removes classification head so model(x) returns [B, D] features
    model = timm.create_model(
        model_id, pretrained=True, num_classes=0, global_pool="token"
    )
    model.eval().to(device)

    cfg = resolve_model_data_config(model)
    transform = create_transform(**cfg, is_training=False)

    embed_dim = int(getattr(model, "num_features", 0))
    if embed_dim <= 0:
        # Fallback: run a tiny dummy forward to infer dim (very rare)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.get("input_size", (3, 224, 224))[1], cfg.get("input_size", (3, 224, 224))[2]).to(device)
            out = model(dummy)
            embed_dim = int(out.shape[-1])

    return model, transform, device, embed_dim


@torch.no_grad()
def encode_image(
    img: Union[str, Image.Image, np.ndarray],
    model_name: str = "dinov2",
    device: Union[str, None] = None,
) -> Tuple[np.ndarray, int]:
    """
    Encode a single image into a L2-normalized 1D feature vector.

    Returns: (feature_vector [D], embed_dim)
    """
    model, transform, device, dim = get_model_and_transform(model_name, device)

    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = Image.fromarray(img[..., :3])
        else:
            raise ValueError("NumPy image must be HxWxC with C>=3")
    elif isinstance(img, Image.Image):
        img = img.convert("RGB")
    else:
        raise TypeError("img must be a path, PIL.Image, or numpy.ndarray")

    x = transform(img).unsqueeze(0).to(device)
    feat = model(x)  # [1, D]
    feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.squeeze(0).cpu().numpy(), dim


@torch.no_grad()
def encode_paths(
    image_paths: List[str],
    model_name: str = "dinov2",
    batch_size: int = 64,
    device: Union[str, None] = None,
) -> Tuple[np.ndarray, int]:
    """
    Encode many image file paths into a 2D array [N, D] of L2-normalized features.

    Returns: (features [N, D], embed_dim)
    """
    if len(image_paths) == 0:
        return np.zeros((0, 0), dtype=np.float32), 0

    model, transform, device, dim = get_model_and_transform(model_name, device)

    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            im = Image.open(p).convert("RGB")
            batch_imgs.append(transform(im))
        x = torch.stack(batch_imgs, dim=0).to(device)
        f = model(x)
        f = torch.nn.functional.normalize(f, dim=-1)
        feats.append(f.cpu())

    feats = torch.cat(feats, dim=0).numpy()
    return feats, dim
