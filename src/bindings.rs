use crate::{DecodeError, Tokenizer};
use pyo3::exceptions::{PyIOError, PyUnicodeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "Tokenizer", module = "bpe_tokenizer")]
struct PyTokenizer {
    tokenizer: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new() -> Self {
        Self {
            tokenizer: Tokenizer::new(),
        }
    }

    #[staticmethod]
    fn train(texts: Vec<String>, target_vocab_size: usize) -> Self {
        Self {
            tokenizer: Tokenizer::train(texts, target_vocab_size),
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.tokenizer.decode(&ids).map_err(decode_error_to_python)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.tokenizer
            .save_to_file(path)
            .map_err(io_error_to_python)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let tokenizer = Tokenizer::load_from_file(path).map_err(io_error_to_python)?;
        Ok(Self { tokenizer })
    }
}

fn decode_error_to_python(error: DecodeError) -> PyErr {
    match error {
        DecodeError::InvalidTokenId(id) => PyValueError::new_err(format!("invalid token ID: {id}")),
        DecodeError::InvalidUtf8(error) => PyUnicodeError::new_err(error.to_string()),
    }
}

fn io_error_to_python(error: std::io::Error) -> PyErr {
    PyIOError::new_err(error.to_string())
}

#[pymodule]
fn bpe_tokenizer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTokenizer>()?;
    Ok(())
}
