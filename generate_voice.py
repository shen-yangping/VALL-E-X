import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.generation import SAMPLE_RATE, generate_audio, preload_models, generate_audio_from_long_text
from scipy.io.wavfile import write as write_wav
import torch
from utils.prompt_making import make_prompt
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # pythonのハッシュベース操作の再現性担保
    np.random.seed(seed)
    torch.manual_seed(seed)  # ネットワーク重みの初期値を固定
    torch.cuda.manual_seed(seed)  # ネットワーク重みの初期値を固定 (GPU)

    # cuDNN: NVIDIAのConvolution高速化ライブラリ．
    torch.backends.cudnn.deterministic = True  # cuDNNによる最適化プロセス固定
    torch.backends.cudnn.benchmark = True  # input_size固定CNNなら高速化が期待できる．iterationごとに変わるならFalseの方が速い．

def generate_voice(input_text, output_wav):
    # set cuda seed
    seed_everything(6861)

    # download and load all models
    preload_models()

    # Use given transcript to make a prompt
    # make_prompt(name="higuchi", audio_prompt_path="./prompts/higuchi_enhance.wav", 
    #             transcript="はい、みなさんこんにちは。今日もチャンネルやすの時間がやってきました。")

    # make_prompt(name="higuchi", audio_prompt_path="prompts/higuchi1.wav")

    # measure inference time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    # generate audio from text
    prompt_file = os.path.join(os.path.dirname(__file__), "customs/higuchi_enhance.npz")
    # audio_array = generate_audio(input_text, prompt=prompt_file,language='ja',  accent="日本語")
    audio_array = generate_audio_from_long_text(input_text, prompt=prompt_file,language='ja',  accent="日本語")

    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end) / 1000, 'sec')

    # save audio to disk
    write_wav(output_wav, SAMPLE_RATE, audio_array)

if __name__ == '__main__':
    input_text = "高いモチベーションで、エネルギッシュに働く人は若手を中心にすごく増えています。ただ、その一方でマインドを切り変えられていない人も一部にはいます。そうした層も含めて、どうやって競争力に変えていくかが今後の課題になりますね。"
    output_wav = "higuchi.wav"
    generate_voice(input_text, output_wav)