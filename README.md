# Anime Speaker Embedding

## Overview

- ECAPA-TDNN model (from [SpeechBrain](https://github.com/speechbrain/speechbrain)) trained on [OOPPEENN/56697375616C4E6F76656C5F44617461736574](https://huggingface.co/datasets/OOPPEENN/56697375616C4E6F76656C5F44617461736574)
- This model is designed for speaker embedding tasks in anime and visual novel contexts.
- **Added Voice Actor (VA) variant** in version 0.2.0, which is less eager to distinguish speakers compared to the default Character (char) variant.

## Features

- Well-suited for **Japanese anime-like** voices, including **non-verbal vocalizations** or **acted voices**
- Also this model works well for *NSFW erotic utterances and vocalizations* such as aegi (喘ぎ) and chupa-sound (チュパ音) which are important culture in Japanese Visual Novel games, while other usual speaker embedding models cannot distinguish such voices of different speakers at all!

## Model Variants

- **char** (default): The model trained to guess character voices, not voice actors, which is eager to distinguish speakers (the model is trained to separate two characters with the same voice actor)
- **va** (added in ver 0.2.0): The model trained on voice actors, not characters, which is less eager to distinguish speakers

Generally, for one fixed character, the **char** model will eagerly separate that character's voice according to styles (resulting in higher variance of embeddings), while the **va** model will try to keep the embeddings of that character's voice similar (resulting in lower variance of embeddings).

## Note

- This model tries to eagerly distinguish speakers, and the values of the cosine similarity between two embeddings of the same speaker are usually lower than other embedding models

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128  # if you want to use GPU
pip install anime_speaker_embedding
```

## Usage

```python
from anime_speaker_embedding import AnimeSpeakerEmbedding
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AnimeSpeakerEmbedding(device=device, variant="char")  # or variant="va" for Voice Actor model
audio_path = "path/to/audio.wav"  # Path to the audio file
embedding = model.get_embedding(audio_path)
print(embedding.shape)  # np.array with shape (192,)
```

See [example.ipynb](example.ipynb) for some usage and visualization examples.

## Comparison with other models

The t-SNE plot of embeddings from some Galgames (not included in the training set!) is shown below.

| Model | Game1 | Game2 | Game3 | Game4 |
|-------|-------|-------|-------|-------|
| [**⭐ VA model**](https://huggingface.co/litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_va_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_va_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_va_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_va_4.jpg) |
| [**⭐ Char model**](https://huggingface.co/litagin/anime_speaker_embedding_ecapa_tdnn_groupnorm) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_char_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_char_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_char_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/anime_char_4.jpg) |
| [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/vanilla_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/vanilla_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/vanilla_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/vanilla_4.jpg) |
| [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resnet34_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resnet34_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resnet34_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resnet34_4.jpg) |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | ![game1](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resemblyzer_1.jpg) | ![game2](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resemblyzer_2.jpg) | ![game3](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resemblyzer_3.jpg) | ![game4](https://raw.githubusercontent.com/litagin02/anime_speaker_embedding/refs/heads/main/assets/resemblyzer_4.jpg) |


- Game1 and Game2 contains NSFW voices, while Game3 and Game4 does not.
- In Game4, Brown and yellow speakers are actually the same character

## Model Details

## Model Architecture

The actual model is [SpeechBrain](https://github.com/speechbrain/speechbrain)'s ECAPA-TDNN **with all BatchNorm layers replaced with GroupNorm**. This is because I encountered a problem with the original BathNorm layers when evaluating the model (maybe some statistics drifted).

### Dataset


#### Char variant

From all the audio files in the [OOPPEENN/56697375616C4E6F76656C5F44617461736574](https://huggingface.co/datasets/OOPPEENN/56697375616C4E6F76656C5F44617461736574) dataset, we filtered out some broken audio files, and exluded the speakers with less than 100 audio files. The final dataset contains:

- train: 6,260,482 audio files, valid: 699,488 audio files, total: 6,959,970 audio files
- 7,357 speakers

#### VA variant

We use [litagin/VisualNovel_Dataset_Metadata](https://huggingface.co/datasets/litagin/VisualNovel_Dataset_Metadata) to obtain the voice actors from the original dataset. We use only characters such that their voice actors are in the vndb. The final dataset contains:

- train: 6,603,080 audio files, valid: 348,034 audio files, total: 6,951,114 audio files
- 989 speakers


### Training process

#### Char variant

- I used [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) as the base model
    - But after some fine-tuning, I replaced all BN with GN, so I don't know how actually the base model effects the final model
    - Also the scaling before fbank is added (`x = x * 32768.0`) (by *mis*advice of ChatGPT), so the original knowledge may not be fully transferred
- First I trained the model on the small subset (the top 100 or 1000 speakers w.r.t. the number of audio files)
- Then I trained the model on the full dataset
- Finally I trained the model on the full dataset with many online augmentations (including reverb, background noise, various filters, etc.)
- At some point, since some characters appear in several games (like FD or same series), I computed the confusion matrix of the model on the validation set, and merged some speakers with high confusion if they are from the same game maker and same character name

#### VA variant

- I used the char variant's backbone as the base model
- Then I just fine-tuned the model, with 0.8 augmentation probability
- I evaluated macro precision / recall / F1 and EER on the validation set, and I adopted the model with the best EER (0.41%). Macro precision, recall, and F1 are 95.97%, 97.83%, and 96.80% respectively.

**The training code will be released in maybe another repo.**
