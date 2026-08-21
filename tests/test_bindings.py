import tempfile
import unittest
from pathlib import Path

from bpe_tokenizer import Tokenizer


class TokenizerTests(unittest.TestCase):
    def test_train_encode_and_decode(self):
        tokenizer = Tokenizer.train(["hello hello"], 260)

        ids = tokenizer.encode("hello")

        self.assertEqual(tokenizer.decode(ids), "hello")
        self.assertEqual(tokenizer.vocab_size(), 260)

    def test_save_and_load(self):
        tokenizer = Tokenizer.train(["banana banana"], 265)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(str(path))
            loaded = Tokenizer.load(str(path))

        self.assertEqual(loaded.encode("banana"), tokenizer.encode("banana"))

    def test_decode_rejects_invalid_token_id(self):
        tokenizer = Tokenizer()

        with self.assertRaises(ValueError):
            tokenizer.decode([2**32 - 1])

    def test_decode_rejects_invalid_utf8(self):
        tokenizer = Tokenizer()

        with self.assertRaises(UnicodeError):
            tokenizer.decode([255])

    def test_load_rejects_missing_file(self):
        with self.assertRaises(OSError):
            Tokenizer.load("file-that-does-not-exist.json")


if __name__ == "__main__":
    unittest.main()
