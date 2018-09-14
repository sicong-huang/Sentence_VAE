## A sequence VAE for sentence generation
This project is a Keras implementation of the paper "Generating Sentences from a Continuous Space" by Bowman et al
### Dependencies
- Python 3.6
- TensorFlow 1.10.0
- Keras 2.2.2
- numpy
- nltk
- scikit-learn
- matplotlib

### Usage
#### Prepare
To download and extract GloVe embedding, in `setup/` run
```bash
./download.sh
```
#### Preprocess
Datasets is included in the repo, to extract vocabulary and preprocess raw data into model-readable form, run
```bash
python build_data.py
```
with optional arguments
#### Train
To train a new pair of encoder and decoder model, run
```bash
python train.py
```
with optional arguments
### Infer
To use trained models to perform inference, run
```bash
python test.py
```
with optional arguments
