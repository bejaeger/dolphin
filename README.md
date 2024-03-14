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
  - [ ] get & transform test data
  - [ ] extract snac codes from data
  - [x] add tokenizer
- [ ] model
  - [x] add snac model
  - [ ] add gpt2 model
- [ ] train loop

## tests to perform
- [ ] do proper text verbalization before tokenization like in [pl-bert](https://github.com/yl4579/PL-BERT/blob/main/text_normalize.py)