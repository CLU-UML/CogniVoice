import os
import sys
import json
import time
import pickle
import argparse
import requests
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from fastchat.model import load_model, get_conversation_template, add_model_args


model_mapping = {
    'gpt3': 'gpt-3', 'gpt3.5': 'gpt-3.5-turbo-0613',
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf', 'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf', 
    'vicuna2-7b': 'lmsys/vicuna-7b-v1.5', 'vicuna2-13b': 'lmsys/vicuna-13b-v1.5', 
    'llama-7b': './llama/hf/7B', 'llama-13b': './llama/hf/13B', 'llama-30b': './llama/hf/30B', 'llama-65b': './llama/hf/65B',
    'vicuna-7b': 'lmsys/vicuna-7b-v1.3', 'vicuna-13b': 'lmsys/vicuna-13b-v1.3', 'vicuna-33b': 'lmsys/vicuna-33b-v1.3',
    'alpaca': './alpaca-7B', 'fastchat-t5': 'lmsys/fastchat-t5-3b-v1.0',
}

template = '''Translate the Chinese into English without extra output: "{}"'''

train = pd.read_parquet('/data/datasets/TAUKADIAL-24/transcription/audio_to_text_w_language_train.parquet')
test = pd.read_parquet('/data/datasets/TAUKADIAL-24/transcription/audio_to_text_w_language_test.parquet')


def get_prompt(args, msg):
    conv = get_conversation_template(model_mapping[args.model_path])
    conv.messages = []
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    conv.system = ''
    prompt = conv.get_prompt().strip()

    return prompt


def main(args=None):
    path = model_mapping[args.model_path]
    model, tokenizer = load_model(
        path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    for name, df in zip(['train', 'test'], [train, test]):
        translation = []
        for lang, text in tqdm(zip(df.language, df.transcribed_text), total=df.shape[0]):
            if lang == 'en':
                translation.append(text)
                continue
            
            msg = template.format(text)
            prompt = get_prompt(args, msg)
            print(prompt)
            outputs = fastchat(prompt, model, tokenizer)
            print(outputs)
            translation.append(outputs)

        df['translation'] = translation
        df.to_parquet(f'/data/datasets/TAUKADIAL-24/transcription/translation_{name}.parquet')


def fastchat(prompt, model, tokenizer):
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    # print('Empty system message')
    # print(f"{conv.roles[0]}: {msg}")
    # print(f"{conv.roles[1]}: {outputs}")

    return outputs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    # parser.add_argument("--system_msg", required=True, type=str, default='default_system_msg')
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)