# Adapted from https://github.com/karpathy/nanoGPT/blob/master/sample.py

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import argparse
from model.gpt2 import GPTConfig, GPT


def main(args: argparse.Namespace) -> None:
    init_from = args.init_from
    out_dir = args.out_dir
    prompt = args.prompt
    num_samples = args.num_samples
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k
    seed = args.seed
    device = args.device
    dtype = args.dtype
    compile = args.compile

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()
    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    print("Running generation")
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                print("----------------")
                print(f"Example #{k}")
                print(f"Prompt: {decode(start_ids)}")
                print(f"Completion: ", end="")                
                for next_token in model.token_generator(x, max_new_tokens, temperature=temperature, top_k=top_k):
                    print(decode([next_token.item()]), end="", flush=True)
                print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using a GPT model.')
    parser.add_argument('--init_from', default='gpt2', help='Either "resume" (from an out_dir) or a GPT-2 variant (e.g., "gpt2"). Default: resume')
    parser.add_argument('--out_dir', default='out', help='Output directory (ignored if init_from is not "resume"). Default: out')
    parser.add_argument('--prompt', default="Why is the universe awesome?", help='Starting prompt for text generation. Can be a file path prefixed with "FILE:". Default: newline')
    parser.add_argument('--num_samples', type=int, default=2, help='Number of samples to generate. Default: 2')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate. Default: 100')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random). Default: 0.8')
    parser.add_argument('--top_k', type=int, default=200, help='Retain only the top_k most likely tokens, clamp others to 0 probability. Default: 200')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed. Default: 1337')
    parser.add_argument('--device', default='cpu', help='Device to use (e.g., "cpu", "cuda", "cuda:0", "cuda:1"). Default: cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16', help='Data type for computations. Default: bfloat16 (if available), otherwise float16')
    parser.add_argument('--compile', action='store_true', help='Use PyTorch 2.0 to compile the model (requires PyTorch 2.0). Default: False')

    main(parser.parse_args())
