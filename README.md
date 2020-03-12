# pyannote.audio hub [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyannote/pyannote-audio/blob/develop/notebooks/introduction_to_pyannote_audio_speaker_diarization_toolkit.ipynb)

This repository serves as a hub for pretrained [pyannote.audio](http://github.com/pyannote/pyannote-audio) models and pipelines, including speech activity detection, speaker change detection, overlapped speech detection, speaker embedding, and speaker diarization.

:warning: Those (free) models are the result of a lot of work from a lot of people, so:
* Everyone - please let us know how you use those models
* Industry - please consider supporting this project financially
* Researcher - please cite relevant papers if you use them in your research:
  - the `pyannote.audio` toolkit
  - research papers describing the used approach
  - datasets used for training models and pipelines

Here is `pyannote.audio` reference if that helps:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```

## Table of contents

- [License](#license)
- [Pipelines](#pipelines)
  - [Usage](#usage)
  - [Speaker diarization](#speaker-diarization)
- [Models](#models)
  - [Usage](#usage-1)
  - [Speech activity detection](#speech-activity-detection)
  - [Speaker change detection](#speaker-change-detection)
  - [Overlapped speech detection](#overlapped-speech-detection)
  - [Speaker embedding](#speaker-embedding)
- [Datasets](#datasets)
  - [DIHARD](#dihard)
  - [AMI](#ami)
  - [VoxCeleb](#voxceleb)
  - [Etape](#etape)


## License

    The MIT License (MIT)

    Copyright (c) 2020- CNRS

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    AUTHORS
    Hervé BREDIN - http://herve.niderb.fr


## Pipelines

A pipeline takes an audio file as input and (usually) returns a [`pyannote.core.Annotation` instance](http://pyannote.github.io/pyannote-core/structure.html#annotation) as output.

### Usage

```python
# load pipeline
import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')

# apply diarization pipeline on your audio file
diarization = pipeline({'audio': '/path/to/your/audio.wav'})

# dump result to disk using RTTM format
with open('/path/to/your/audio.rttm', 'w') as f:
    diarization.write_rttm(f)
  
# iterate over speech turns
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')
# Speaker "A" speaks between t=0.2s and t=1.4s.
# Speaker "B" speaks between t=2.3s and t=4.8s.
# Speaker "A" speaks between t=5.2s and t=8.3s.
# Speaker "C" speaks between t=8.3s and t=9.4s.
# ...
```

### Speaker diarization

|   | Pipeline               | Models used internally                       | Development set
|---|------------------------|----------------------------------------------|-----------------
| ✅ | `dia` or `dia_dihard`  | {`sad_dihard`, `scd_dihard`, `emb_voxceleb`} | `DIHARD.custom.dev`
| ✅ | `dia_ami`              | {`sad_ami`, `scd_ami`, `emb_ami`}            | `AMI.dev`
| ❌ | `dia_etape`            | {`sad_etape`, `scd_etape`, `emb_etape`}      | `Etape.dev`

Pipelines marked with ❌ are not available yet but will be released at some point.

### Citation

```bibtex
@inproceedings{Yin2018,
  Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
  Title = {{Neural Speech Turn Segmentation and Affinity Propagation for Speaker Diarization}},
  Booktitle = {{19th Annual Conference of the International Speech Communication Association, Interspeech 2018}},
  Year = {2018},
  Month = {September},
  Address = {Hyderabad, India},
}
```

## Models

A model takes an audio file as input and returns a [`pyannote.core.SlidingWindowFeature` instance](http://pyannote.github.io/pyannote-core/reference.html#pyannote.core.SlidingWindowFeature) as output, that contains the raw output of the underlying neural network. 

### Usage

```python
# load model
import torch
model = torch.hub.load('pyannote/pyannote-audio', 'sad')

# apply model on your audio file
raw_scores = model({'audio': '/path/to/your/audio.wav'})
```

Most models can also be loaded as pipelines, using the `pipeline=True` option:

```python
# load model and wrap it in a pipeline
import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'sad', pipeline=True)

# apply speech activity detection pipeline on your audio file
speech_activity_detection = pipeline({'audio': '/path/to/your/audio.wav'})

# dump result to disk using RTTM format
with open('/path/to/your/audio.sad.rttm', 'w') as f:
    speech_activity_detection.write_rttm(f)

for speech_region in speech_activity_detection.get_timeline():
    print(f'There is speech between t={speech_region.start:.1f}s and t={speech_region.end:.1f}s.')
# There is speech between t=0.2s and t=1.4s.
# There is speech between t=2.3s and t=4.8s.
# There is speech between t=5.2s and t=8.3s.
# There is speech between t=8.3s and t=9.4s.
# ...
```

### Speech activity detection

|   | Model                | Training set        | Development set
|---|----------------------|---------------------|-----------------
| ✅ | `sad` or `sad_dihard`| `DIHARD.custom.trn` | `DIHARD.custom.dev`
| ✅ |`sad_ami`             | `AMI.trn`           | `AMI.dev`
| ❌ |`sad_etape`           | `Etape.trn`         | `Etape.dev`

Models marked with ❌ are not available yet but will be released at some point.

#### Citation

```bibtex
@inproceedings{Lavechin2020,
    author = {Marvin Lavechin and Marie-Philippe Gill and Ruben Bousbib and Herv\'{e} Bredin and Leibny Paola Garcia-Perera},
    title = {{End-to-end Domain-Adversarial Voice Activity Detection}},
    year = {2020},
    url = {https://arxiv.org/abs/1910.10655},
}
```

### Speaker change detection

|   | Model                 | Training set        | Development set
|---|-----------------------|---------------------|-----------------
| ✅ | `scd` or `scd_dihard` | `DIHARD.custom.trn` | `DIHARD.custom.dev`
| ✅ |`scd_ami`              | `AMI.trn`           | `AMI.dev`
| ❌ |`scd_etape`            | `Etape.trn`         | `Etape.dev`

Models marked with ❌ are not available yet but will be released at some point.


#### Citation

```bibtex
@inproceedings{Yin2017,
  Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
  Title = {{Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks}},
  Booktitle = {{Interspeech 2017, 18th Annual Conference of the International Speech Communication Association}},
  Year = {2017},
  Month = {August},
  Address = {Stockholm, Sweden},
}
```

### Overlapped speech detection

|   | Model                | Training set        | Development set
|---|----------------------|---------------------|-----------------
| ✅ |`ovl` or `ovl_dihard` | `DIHARD.custom.trn` | `DIHARD.custom.dev`
| ✅ |`ovl_ami`             | `AMI.trn`           | `AMI.dev`
| ❌ |`ovl_etape`           | `Etape.trn`         | `Etape.dev`

Models marked with ❌ are not available yet but will be released at some point.


#### Citation

```bibtex
@inproceedings{Bullock2020,
  Title = {{Overlap-aware diarization: resegmentation using neural end-to-end overlapped speech detection}},
  Author = {{Bullock}, Latan{\'e} and {Bredin}, Herv{\'e} and {Garcia-Perera}, Leibny Paola},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```

### Speaker embedding

Speaker embedding models cannot be loaded as pipelines.

```python
# load model
import torch
model = torch.hub.load('pyannote/pyannote-audio', 'emb')

print(f'Embedding has dimension {model.dimension:d}.')
# Embedding has dimension 512.

# extract speaker embedding on the whole file using built-in sliding window
import numpy as np
from pyannote.core import Segment
embedding = model({'audio': '/path/to/your/audio.wav'})
for window, emb in embedding:
    assert isinstance(window, Segment)
    assert isinstance(emb, np.ndarray)    

# extract speaker embedding of an excerpt
excerpt1 = Segment(start=2.3, end=4.8)
emb1 = model.crop({'audio': '/path/to/your/audio.wav'}, excerpt1)
assert isinstance(emb1, np.ndarray)

# compare speaker embedding
from scipy.spatial.distance import cdist
excerpt2 = Segment(start=5.2, end=8.3)
emb2 = model.crop({'audio': '/path/to/your/audio.wav'}, excerpt2)
distance = cdist(np.mean(emb1, axis=0, keepdims=True), 
                 np.mean(emb2, axis=0, keepdims=True), 
                 metric='cosine')[0, 0]
```

|    | Model                 | Training set                             | Development set
|----|-----------------------|------------------------------------------|-----------------
| ✅ |`emb` or `emb_voxceleb` | `VoxCeleb1.custom.trn` ⋃ `VoxCeleb2.trn` | `VoxCeleb1.custom.dev`
| ✅ |`emb_ami`               | `AMI.trn`                                | `AMI.dev`


#### Citation

```bibtex
@inproceedings{Bredin2017,
    author = {Herv\'{e} Bredin},
    title = {{TristouNet: Triplet Loss for Speaker Turn Embedding}},
    booktitle = {42nd IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017},
    year = {2017},
    url = {http://arxiv.org/abs/1609.04301},
}
```

## Datasets

### DIHARD

The single-channel subset of the [DIHARD dataset](#citations) contains approximately 47 hours of audio:
- `DIHARD.dev` | a development set of about 24 hours (192 files)
- `DIHARD.tst` | a test set of 22 hours (194 files)

Since no training set is provided, we split the official development set (`DIHARD.dev`) in two parts: 
- `DIHARD.custom.trn` | two third (16 hours, 126 files) serve as training set; 
- `DIHARD.custom.dev` | the other third (8 hours, 66 files) serves as development set. 

The test set has **not** been used to train any of the provided models and pipelines. 

#### Citation

```bibtex
@inproceedings{Ryant2019,
  author={Neville Ryant and Kenneth Church and Christopher Cieri and Alejandrina Cristia and Jun Du and Sriram Ganapathy and Mark Liberman},
  title={{The Second DIHARD Diarization Challenge: Dataset, Task, and Baselines}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={978--982},
  doi={10.21437/Interspeech.2019-1268},
  url={http://dx.doi.org/10.21437/Interspeech.2019-1268}
}
```

### AMI

AMI (headset mix) is a subset of the [AMI corpus](#citations) that consists of summed recordings of spontaneous speech of mainly four speakers. We use the [official split](http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml)

- `AMI.trn` | a training set of about 70 hours
- `AMI.dev` | a development set of about 14 hours
- `AMI.tst` | a test set of about 14 hours

The test set has **not** been used to train any of the provided models and pipelines. 

#### Citation

```bibtex
@article{Carletta2007,  
  Title = {{Unleashing the killer corpus: experiences in creating the multi-everything AMI Meeting Corpus}},
  Author = {Carletta, Jean},
  Journal = {Language Resources and Evaluation},
  Volume = {41},
  Number = {2},
  Year = {2007},
}
```

### VoxCeleb

VoxCeleb contains audio recordings of a large number of different speakers, and is divided into three smaller subsets:
- `VoxCeleb1.dev` | recordings of xx speakers
- `VoxCeleb1.tst` | recordings of 40 speakers
- `VoxCeleb2.trn` | recordings of xx speakers

Since no training set is provided for VoxCeleb 1, we split the official development into two parts such that `VoxCeleb1.dev` = `VoxCeleb1.custom.trn` ⋃ `VoxCeleb1.custom.dev`:
- `VoxCeleb1.custom.trn` | all `VoxCeleb1.dev` speakers but the 41 whose name starts with U, V, or W.
- `VoxCeleb1.custom.dev` | 41 speakers of `VoxCeleb1.dev` whose name starts with U, V, or W.

Both `VoxCeleb1.custom.trn` and `VoxCeleb2.trn` were used to train models, while `VoxCeleb1.custom.dev` was used for selecting the best epoch.

VoxCeleb 1 test set (`VoxCeleb1.tst`) has **not** been used to train any of the provided models and pipelines.

#### Citation

```bibtex
@article{Nagrani19,
  Author = {Arsha Nagrani and Joon~Son Chung and Weidi Xie and Andrew Zisserman},
  Title = {{Voxceleb: Large-scale speaker verification in the wild},
  Journal = {Computer Science and Language},
  Year = {2019},
}
```

### Etape

```bibtex
@inproceedings{etape,
  title = {{The ETAPE Corpus for the Evaluation of Speech-based TV Content Processing in the French Language}},
  Author = {Gravier, Guillaume and Adda, Gilles and Paulson, Niklas and Carr{\'e}, Matthieu and Giraudel, Aude and Galibert, Olivier},
  Booktitle = {{Proc. LREC 2012}},
  Year = {2012},
}
```

#### Citation
