# dolphin

## specs
- use [snac](https://github.com/hubertsiuzdak/snac) speech audio codec
- train gpt-2 to predict audio tokens auto-regressively

## install
1. clone repo
2. update submodules `git submodule update --init --recursive`
3. download snac model using git lfs `cd model/codec/snac/snac_24khz` & `git lfs pull` 
4. install dependencies `pip install -r requirements.txt`

## roadmap
- [ ] data
  - [x] add normalizer & tokenizer
  - [ ] add interface audio -> codes
  - [ ] code to prepare data (audio, text, snac codes)
  - [ ] get & transform test dataset (500h)
- [ ] model
  - [x] add snac model
  - [x] add gpt2 model
  - [ ] interface for combining text & audio code as input to sequence model
  - [ ] add prediction layer (classification heads after sequence model)
  - [ ] interface for dealing with output from sequence model (loss computation)
- [ ] train loop
- [ ] api & app to synthesize speech

## future projects
- [ ] use RWKV instead of gpt2 https://github.com/BlinkDL/RWKV-LM
- [ ] add speaker style conditioning
