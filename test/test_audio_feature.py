import unittest

import numpy as np

from data.audio import AudioFeature, AudioFeatures

INIT_ARGUMENTS = {
    AudioFeatures.SNAC_CODEC: {
        "feature": AudioFeatures.SNAC_CODEC.value,
        "model_path": "model/codec/snac/snac_24khz/pytorch_model.bin",
        "config_path": None,
        "device": "cpu",
    }
}

AUDIO_PATH = "test/resource/voice_sample_tiny.wav"


class TestAudioFeature(unittest.TestCase):
    def test_snac_codec(self):
        feature = AudioFeature.create(**INIT_ARGUMENTS[AudioFeatures.SNAC_CODEC])

        initial_codes = [np.array([[5]]), np.array([[3, 4]]), np.array([[0, 1, 2, 3]])]
        target_codes = [np.array([[5, 5, 5, 5]]), np.array([[3, 3, 4, 4]]), np.array([[0, 1, 2, 3]])]

        processed_codes = feature.expand_codes(initial_codes)
        self.assertTrue(np.equal(processed_codes, target_codes).all(), "Error: expand codes")

        processed_codes = feature.collapse_codes(processed_codes)
        for i in range(len(processed_codes)):
            self.assertTrue(np.equal(processed_codes[i], np.array(initial_codes[i])).all(), "Error: collapse codes")

        codes = feature.encode(AUDIO_PATH)

        self.assertTrue(len(codes) == feature.NUM_HIERARCHIES, "Error: number of hierarchies")


if __name__ == '__main__':
    unittest.main()
