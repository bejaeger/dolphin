import unittest
import subprocess
import os


class TestApps(unittest.TestCase):
    def _test_app(self, cmd: str, app_name: str):
        print(f"Running command: `{cmd}`")
        try:
            subprocess.run(cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            self.fail(f"Error occurred while running {app_name}: {e.stderr.decode('utf-8')}")

    def test_tokenize_app(self):
        cmd = "python3.9 -m app.tokenize"
        self._test_app(cmd, "app.tokenize")

    def test_encode_decode_app(self):
        model_path = 'model/codec/snac/snac_24khz/pytorch_model.bin'
        input_path = 'resource/dataset/example_audio/voice_sample_tiny.wav'
        output_path = 'out/test.wav'

        cmd = f"python3.9 -m app.encode_decode -m {model_path} -i {input_path} -o {output_path}"
        self._test_app(cmd, "app.encode_decode")

        self.assertTrue(os.path.isfile(output_path), "Output file was not created")

    def test_generate_text_app(self):
        cmd = "python3.9 -m app.generate_text --num_samples 1 --max_new_tokens 50"
        self._test_app(cmd, "app.generate_text")


if __name__ == '__main__':
    unittest.main()