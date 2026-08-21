# BPE Tokenizer

This project contains a byte-level BPE tokenizer in Rust.

The tokenizer can train on text. It can encode, decode, save, and load data.

## Rust example

```rust
use bpe_tokenizer::Tokenizer;

fn main() -> std::io::Result<()> {
    let tokenizer = Tokenizer::train(["hello hello"], 260);
    let ids = tokenizer.encode("hello");
    let text = tokenizer.decode(&ids).unwrap();

    tokenizer.save_to_file("tokenizer.json")?;
    let loaded = Tokenizer::load_from_file("tokenizer.json")?;

    assert_eq!(loaded.decode(&loaded.encode(&text)).unwrap(), text);
    Ok(())
}
```

## Python example

Install the package in a virtual environment:

```bash
python -m pip install maturin
maturin develop --release
```

Use the tokenizer:

```python
from bpe_tokenizer import Tokenizer

tokenizer = Tokenizer.train(["hello hello"], 260)
ids = tokenizer.encode("hello")
text = tokenizer.decode(ids)
```
