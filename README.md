# StyleTTS-VC for Japanese

### Overview
StyleTTS-VC model is modified from [the repo](https://github.com/yl4579/StyleTTS-VC) for Japanese.

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/QuyAnh2005/StyleTTS-VC-Japanese.git
cd StyleTTS-VC-Japanese
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```
4. Dataset 
Dataset is downloaded from
   - [100 speakers](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
   - [1 speaker](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
   
and locate at `dataset` folder.
     
## Preprocessing

The pretrained text aligner and pitch extractor models are provided under the `Utils` folder. Both the text aligner and pitch extractor models are trained with melspectrograms preprocessed using [meldataset.py](https://github.com/yl4579/StyleTTS-VC/blob/main/meldataset.py). 

You can edit the [meldataset.py](meldataset.py) with your own melspectrogram preprocessing, but the provided pretrained models will no longer work. You will need to train your own text aligner and pitch extractor with the new preprocessing. 

The code for training new text aligner model is available [here](https://github.com/yl4579/AuxiliaryASR) and that for training new pitch extractor models is available [here](https://github.com/yl4579/PitchExtractor).

The data list format needs to be `filename.wav|transcription|speaker`, see [val_list.txt](Data/val_list.txt) as an example. The speaker information is needed in order to perform speaker-dependent adversarial training. 

To convert data into phonemes before training. Run
```bash
python preprocess.py
```

## Training
First stage training:
```bash
python train_first.py --config_path ./Configs/config.yml
```
Second stage training:
```bash
python train_second.py --config_path ./Configs/config.yml
```
Pretrained models are available at [here.](https://drive.google.com/drive/folders/1s-Mrpu8dmfrF_6hMeL7cK1cfxAa1d4NG?usp=sharing)

## Inference

Please refer to [inference.ipynb](/Demo/Inference.ipynb) for details. 

The pretrained StyleTTS-VC on Japanese dataset and Hifi-GAN on LibriTTS corpus in 24 kHz can be downloaded at [StyleTTS-VC Link](https://drive.google.com/file/d/1dB-G-JT3Jd9WoY9FShh8M-qrNMwFnd9a/view?usp=sharing) and [Hifi-GAN Link](https://drive.google.com/file/d/1RDxYknrzncGzusYeVeDo38ErNdczzbik/view?usp=sharing). 

Please unzip to `Models` and `Vocoder` respectivey and run each cell in the notebook. 

Run [app.py](app.py) to see demo using gradio.

