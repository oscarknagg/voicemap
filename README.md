# one-shot-speaker-identification

Identifying speakers with deep learning.

## Instructions
#### Requirements
Make a new virtualenv and install requirements from `requirements.txt` with
```
pip install -r requirements.txt
```
This project was written in Python 2.7.12 so I cannot guarantee it works on
any other version.

#### Run tests

```
python -m unittest tests.tests
```

#### Data
Get training data here: http://www.openslr.org/12
- train-clean-100.tar.gz
- train-clean-360.tar.gz
- dev-clean.tar.gz

Place the unzipped training data into the `data/` folder so the file structure is as follows:
```
data/
    LibriSpeech/
        dev-clean/
        train-clean-100/
        train-clean-360/
        SPEAKERS.TXT
```

Please use the `SPEAKERS.TXT` supplied in the repo as I've made a few corrections to the one found at openslr.org.




# Notes

### Datasets

- LibriSpeech
- Mozilla Common Voice
- Google Audio Commands
- Speakers in the wild
- https://catalog.ldc.upenn.edu/LDC2017S06 (EXPENSIVE)
- VoxCeleb https://arxiv.org/pdf/1706.08612.pdf

### Papers

https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
http://danielpovey.com/files/2015_asru_tdnn_ubm.pdf
https://www.sri.com/sites/default/files/publications/final2c_the_2016_speakers_in_the_wild_speaker_recognition_evaluation_3.pdf
