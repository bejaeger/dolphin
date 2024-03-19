# Write downloader for different datasets that are available online (libri tts, ljspeech, self generated etc.)

# TODO Next (start with a small example dataset and push through pipeline)

# Save in following folder structure:
# 
# dataset/
# ├── dataset.jsonl  # The main dataset file in JSONLines format
# ├── speaker_embeds/
# │   ├── speaker_0.wav
# │   ├── speaker_1.wav
# │   ├── ...
# │   └── speaker_9.wav
# └── audio/
#     ├── speaker_0/
#     │   ├── audio_file_1.wav
#     │   ├── audio_file_2.wav
#     │   ├── ...
#     ├── speaker_1/
#     │   ├── ...
#     ├── ...
#     └── speaker_9/
#         ├── ...
