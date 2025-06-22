from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

REPO_ID_DICT = {
    "char": "litagin/anime_speaker_embedding_ecapa_tdnn_groupnorm",
    "va": "litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm",
}
FILE_NAME = "embedding_model.pth"
REVISION = "main"


def _get_ckpt(ckpt_path: Union[str, Path, None], variant: Union[str, None]) -> Path:
    """Resolve local checkpoint path; downloads from HF Hub if needed."""
    if ckpt_path is None:
        if variant is None:
            variant = "char"
        if variant not in REPO_ID_DICT:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: {list(REPO_ID_DICT.keys())}"
            )
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID_DICT[variant], filename=FILE_NAME, revision=REVISION
        )
    return Path(ckpt_path)


def swap_bn_to_gn(model: nn.Module, num_groups: int = 32) -> None:
    """
    Replace all torch.nn.BatchNorm1d with GroupNorm(num_groups, C).
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            C = child.num_features
            assert C % num_groups == 0
            gn = nn.GroupNorm(num_groups, C, affine=True)
            # Copy γ and β
            gn.weight.data.copy_(child.weight.data)
            gn.bias.data.copy_(child.bias.data)
            setattr(model, name, gn.to(child.weight.device))
        else:
            swap_bn_to_gn(child, num_groups=num_groups)


class AnimeSpeakerEmbedding(nn.Module):
    def __init__(
        self,
        variant: Union[str, None] = None,
        *,
        n_mels: int = 80,
        lin_neurons: int = 192,
        channels: list[int] = [1024, 1024, 1024, 1024, 3072],
        kernel_sizes: list[int] = [5, 3, 3, 3, 1],
        num_groups: int = 32,
        sr: int = 16_000,
        ckpt_path: Union[str, Path | None] = None,
        device: Union[str, None] = None,
    ) -> None:
        super().__init__()
        self.backbone = ECAPA_TDNN(
            input_size=n_mels,
            lin_neurons=lin_neurons,
            channels=channels,
            kernel_sizes=kernel_sizes,
        )
        swap_bn_to_gn(self.backbone, num_groups=num_groups)
        self.fbank = Fbank(sample_rate=sr, n_mels=n_mels)

        ckpt_path = _get_ckpt(ckpt_path, variant=variant)
        print(f"Loading checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(state, strict=False)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = device
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time)
        max_abs = torch.max(torch.abs(x))
        if max_abs > 1.0:
            x = x / max_abs
        # NOTE: The below scaling (32768) is not used in the original pretrained model.
        # But I wrongly used it in the training script, so I should keep it.
        x = x * 32768.0
        x = self.fbank(x)  # (batch, n_frames, n_mels)
        x = self.backbone(x)  # (batch, 1, 192)
        return x

    @torch.inference_mode()
    def get_embedding(self, audio_path: Union[Path, str]) -> np.ndarray:
        """
        Extract embedding from an audio file.
        Args:
            audio_path (Path): Path to the audio file.
        Returns:
            np.ndarray: Normalized embedding vector of shape (192,).
        """
        wav, _ = librosa.load(audio_path, sr=16_000, mono=True)
        wav_torch = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        embedding = self(wav_torch)  # (1, 1, 192)
        embedding = embedding.squeeze(0).squeeze(0)  # (192,)
        embedding = F.normalize(embedding, dim=0)  # L2 normalization
        return embedding.cpu().numpy()
