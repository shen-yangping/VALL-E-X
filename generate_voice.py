import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import torch
from utils.prompt_making import make_prompt

def generate_voice(input_text, output_wav):
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
    prompt_file = os.path.join(os.path.dirname(__file__), "customs/higuchi.npz")
    audio_array = generate_audio(input_text,language='ja' ,prompt=prompt_file, accent="日本語")

    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end) / 1000, 'sec')

    # save audio to disk
    write_wav(output_wav, SAMPLE_RATE, audio_array)

    # play text in notebook
    # Audio(audio_array, rate=SAMPLE_RATE)

if __name__ == '__main__':
    input_text = "今日もチャンネルやすの時間がやってきました。"
    output_wav = "higuchi.wav"
    generate_voice(input_text, output_wav)