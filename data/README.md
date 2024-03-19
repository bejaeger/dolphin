# Steps to get to data loader for training

1. `app/download.py`: download audio dataset + transcription and put in right folder structure
2. `app/process_and_filter.py`: select quality data, split audio, run automatic transcription, potential speech enhancement
4. `app/prepare.py`: transform dataset (extract audio tokens, prepare speaker conditioning file/embedding, tokenize data, save metadata like tokenizer, audio_feature used)


# Classes
1. `transformer.py`: dataset transformer used in `prepare.py`
2. `dataset.py`: loads transformed data automatically choosing tokenizer & audio_feature
3. `data_loader.py`: data loader for training
