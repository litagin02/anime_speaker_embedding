# Anime Speaker Embedding

## Overview

- ECAPA-TDNN model (from [SpeechBrain](https://github.com/speechbrain/speechbrain)) trained on [OOPPEENN/VisualNovel_Dataset](https://huggingface.co/datasets/OOPPEENN/VisualNovel_Dataset) (a.k.a. Galgame_Dataset)
- This model is designed for speaker embedding tasks in anime and visual novel contexts.

## Features

- This model are well-suited for **anime and visual novel** voices, including non-verbal vocalizations or short utterances.
- Also this model can deal with **NSFW** utterances and vocalizations such as aegi (喘ぎ) and chupa-sound (チュパ音), while other usual speaker embedding models cannot distinguish such voices of different speakers

## Comparison with other models

The t-SNE plot of embeddings from some Galgames (not included in the training set!) is shown below.

| model | Game1 | Game2 | Game3 | Game4 |
| --- | --- | --- | --- | --- |
| [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/vanilla_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/vanilla_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/vanilla_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/vanilla_4.jpg) |
| [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resnet34_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resnet34_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resnet34_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resnet34_4.jpg) |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resemblyzer_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resemblyzer_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resemblyzer_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resemblyzer_4.jpg) |
| [**this model**](https://huggingface.co/litagin/anime_speaker_embedding) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_4.jpg) |

## Installation

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128  # If you use GPU torch
uv pip install anime_speaker_embedding
```

## Usage

```python
from anime_speaker_embedding.model import AnimeSpeakerEmbedding
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AnimeSpeakerEmbedding(device=device)
audio_path = "path/to/audio.wav"  # Path to the audio file
embedding = model.get_embedding(audio_path)
print(embedding.shape)  # np.array with shape (192,)
```

See [example.ipynb](example.ipynb) for some usage and visualization examples.

## Model Details

The actual model is ECAPA-TDNN **with all BatchNorm layers replaced with GroupNorm**. This is because I encountered a problem with the original BathNorm layers when evaluating the model (maybe some statistics drifted).

The model structure is in [model.py](model.py).

### Dataset

From all the audio files in the [OOPPEENN/VisualNovel_Dataset](https://huggingface.co/datasets/OOPPEENN/VisualNovel_Dataset) dataset, we filtered out some broken audio files, and exluded the speakers with less than 100 audio files. The final dataset contains:

- 6,164,667 audio files (hours will be calculated later)
- 6,235 speakers

### Training process

- I used [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) as the base model (but after some fine-tuning, I replaced all BN with GN, so I don't know how actually the base model effects the final model)
- First I trained the model on the small subset (the top 100 or 1000 speakers w.r.t. the number of audio files)
- Then I trained the model on the full dataset
- Finally I trained the model on the full dataset with online augmentations (including reverb, noise, etc.)

**The training code will be released in maybe another repo.**
