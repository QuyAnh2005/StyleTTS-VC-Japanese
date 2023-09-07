# load packages
import random
import yaml
import gradio as gr
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import scipy.io.wavfile as wavfile
import tempfile
from scipy.io import wavfile

from models import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(ref_path, model):
    wav, sr = librosa.load(ref_path, sr=24000)
    audio, index = librosa.effects.trim(wav, top_db=25)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref = model.style_encoder(mel_tensor.unsqueeze(1))

    return ref.squeeze(1), audio

# load hifi-gan
import sys
sys.path.insert(0, "./Demo/hifi-gan")
import glob
import os
import json
import torch
from attrdict import AttrDict
from vocoder import Generator

h = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

cp_g = scan_checkpoint("Vocoder/LibriTTS/", 'g_')

config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

device = torch.device(device)
generator = Generator(h).to(device)

state_dict_g = load_checkpoint(cp_g, device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()


# load StyleTTS
model_path = "./Models/JP/epoch_2nd_00108.pth"
model_config_path = "./Models/JP/config.yml"

config = yaml.safe_load(open(model_config_path))
# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

model = build_model(Munch(config['model_params']), text_aligner, pitch_extractor)

params = torch.load(model_path, map_location='cpu')
params = params['net']
for key in model:
    if key in params:
        if not "discriminator" in key:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key])
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]


def voice_conversion(source_audio, ref_audio):
    # Write the audio data to the WAV file
    sr, wav = source_audio
    source_path = './tmp/source.wav'
    wavfile.write(source_path, sr, wav)
    sr, wav = ref_audio
    ref_path = './tmp/reference.wav'
    wavfile.write(ref_path, sr, wav)

    audio, source_sr = librosa.load(source_path, sr=24000)
    audio, index = librosa.effects.trim(audio, top_db=25)
    audio = audio / np.max(np.abs(audio))
    audio.dtype = np.float32
    source = preprocess(audio).to(device)

    with torch.no_grad():
        mel_input_length = torch.LongTensor([source.shape[-1]])
        asr = model.mel_encoder(source)
        F0_real, _, F0 = model.pitch_extractor(source.unsqueeze(1))
        real_norm = log_norm(source.unsqueeze(1)).squeeze(1)

        ref, _ = compute_style(ref_path, model)
        out = model.decoder(asr, F0_real.unsqueeze(0), real_norm, ref.squeeze(1))
        c = out.squeeze()
        y_g_hat = generator(c.unsqueeze(0))
        y_out = y_g_hat.squeeze()
        y_out = y_out.cpu().numpy()

    # Save the converted audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, 24000, y_out)
        audio_file = f.name

    return audio_file


# Create the input and output interfaces for Gradio
source_audio_interface = gr.inputs.Audio(label='Original-Speaker Audio')
target_audio_interface = gr.inputs.Audio(label='Reference-Speaker Audio')
output_audio_interface = gr.outputs.Audio(label='Output Audio', type='filepath')

# Create the Gradio interface
gr.Interface(
    fn=voice_conversion,
    inputs=[source_audio_interface, target_audio_interface],
    outputs=output_audio_interface,
    title='Voice Conversion for Japanese Demo').queue().launch(share=True)
