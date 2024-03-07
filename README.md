# dolphin

## specs
- use [snac](https://github.com/hubertsiuzdak/snac) speech audio codec
- train gpt-2 to predict audio tokens auto-regressively

## install
1. clone repo
2. update submodules `git submodule update --init --recursive`
3. download snac model using git lfs `cd model/snac/snac_24khz` & `git lfs pull` 
4. install dependencies `pip install -r requirements.txt`

## roadmap
- [ ] data
  - [ ] get & transform test data
  - [ ] extract snac audio tokens from data
- [ ] model
  - [x] add snac model
  - [ ] add gpt2 model
- [ ] train loop
