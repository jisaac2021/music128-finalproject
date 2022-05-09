# music128-finalproject

Structure of the Repo

The docs folder has our website code including Javascript code to interpolate between different audio files in realtime.
experiments/duet_dataloader has the dataloading code for the genre classifier project. (We also modified the ss-vq-vae itself to output styles, but I don't think that code is currently in the repository).
The gm-vae branch has the GM-VAE modifications (stochastic sampling).
The RNN code is in experiments/duet_dataloader and src/ss_vq_vae
How to use the repo
We've added two new flags to the python -m ss_vq_vae.models.vqvae_oneshot command described in the original README below:

encode_content will output just the content (you need to pass in a path to a file that contains a list of lines, each line needs a path to a file to encode content of)
The model path is currently hardcoded to "/datasets/duet/ssvqvae_model_state.pt" on line 307 of vqvae_oneshot.py -- you'll need to change it or run on CSUA
interpolate will output the 10 interpolation files for each audio file pair. It takes a path to pairs of files (each pair is tab separated I think, and then lines between pairs)
interpolate also takes a --model flag with the model path
For training, it's the same as the original.

Overview of the project and results
https://docs.google.com/document/d/1t5M4AntR1DhEWCarxSKBkeX1C3524PaDhqQKgubu9jM/edit?usp=sharing

Self-Supervised VQ-VAE for One-Shot Music Style Transfer
This is the code repository for the ICASSP 2021 paper Self-Supervised VQ-VAE for One-Shot Music Style Transfer by Ondřej Cífka, Alexey Ozerov, Umut Şimşekli, and Gaël Richard.

Copyright 2020 InterDigital R&D and Télécom Paris.

Links
🔬 Paper preprint [pdf]
🎵 Supplementary website with audio examples
🎤 Demo notebook
🧠 Trained model parameters (212 MB)

Contents
src – the main codebase (the ss-vq-vae package); install with pip install ./src; usage details below
data – Jupyter notebooks for data preparation (details below)
experiments – model configuration, evaluation, and other experimental stuff
Setup
pip install -r requirements.txt
pip install ./src
Usage
To train the model, go to experiments, then run:

python -m ss_vq_vae.models.vqvae_oneshot --logdir=model train
This is assuming the training data is prepared (see below).

To run the trained model on a dataset, substitute run for train and specify the input and output paths as arguments (use run --help for more information). Alternatively, see the colab_demo.ipynb notebook for how to run the model from Python code.

Datasets
Each dataset used in the paper has a corresponding directory in data, containing a Jupyter notebook called prepare.ipynb for preparing the dataset:

the entire training and validation dataset: data/comb; combined from LMD and RT (see below)
Lakh MIDI Dataset (LMD), rendered as audio using SoundFonts
the part used as training and validation data: data/lmd/audio_train
the part used as the 'artificial' test set: data/lmd/audio_test
both require downloading the raw data and pre-processing it using data/lmd/note_seq/prepare.ipynb
the following SoundFonts are required (available here and here): FluidR3_GM.sf2, TimGM6mb.sf2, Arachno SoundFont - Version 1.0.sf2, Timbres Of Heaven (XGM) 3.94.sf2
RealTracks (RT) from Band-in-a-Box UltraPAK 2018 (not freely available): data/rt
Mixing Secrets data
the 'real' test set: data/mixing_secrets/test
the set of triplets for training the timbre metric: data/mixing_secrets/metric_train
both require downloading and pre-processing the data using data/mixing_secrets/download.ipynb
