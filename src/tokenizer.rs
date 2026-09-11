use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{Error, ErrorKind};
use rayon::prelude::*;

const FILE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct TokenizerFile {
    version: u32,
    vocab: Vec<Vec<u8>>,
    merges: Vec<MergeFile>,
}

#[derive(Serialize, Deserialize)]
struct MergeFile {
    left: u32,
    right: u32,
    merged_id: u32,
    rank: usize,
}

#[derive(Debug)]
pub enum DecodeError {
    InvalidTokenId(u32),
    InvalidUtf8(std::string::FromUtf8Error),
}

#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub vocab: Vec<Vec<u8>>,
    pub token_to_id: HashMap<Vec<u8>, u32>,
    pub merges: HashMap<(u32, u32), u32>,
    pub ranks: HashMap<(u32, u32), usize>,
}

impl Tokenizer {
    pub fn new() -> Self {
        let (vocab, token_to_id) = Self::build_initial_vocab();
        let merges = HashMap::new();
        let ranks = HashMap::new();
        Self {
            vocab,
            token_to_id,
            merges,
            ranks,
        }
    }

    pub fn from_parts(
        vocab: Vec<Vec<u8>>,
        token_to_id: HashMap<Vec<u8>, u32>,
        merges: HashMap<(u32, u32), u32>,
        ranks: HashMap<(u32, u32), usize>,
    ) -> Self {
        Self {
            vocab,
            token_to_id,
            merges,
            ranks,
        }
    }

    // Input: multiple training texts and a target vocabulary size.
    // Output: a tokenizer whose vocabulary starts with 256 byte tokens and
    // grows by repeatedly merging the most frequent adjacent pair.
    // Example: training ["aaaa"] with target 257 should learn [a, a] -> "aa".
    // Important edge cases:
    //   - Do not create pairs across separate input texts.
    //   - Stop if there are no adjacent pairs left.
    //   - A target below 256 still requires the complete byte vocabulary.
    pub fn train<I, S>(texts: I, target_vocab_size: usize) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut tokenizer = Self::new();

        let mut sequences: Vec<Vec<u32>> = texts
            .into_iter()
            .map(|text| {
                text.as_ref()
                    .as_bytes()
                    .iter()
                    .map(|&byte| byte as u32)
                    .collect()
            })
            .collect();

        while tokenizer.vocab_size() < target_vocab_size {
            // Parallel counting: each thread counts its chunk, then we merge the HashMaps
            let counts = sequences.par_iter().fold(
                HashMap::new,
                |mut acc: HashMap<(u32, u32), usize>, seq| {
                    for pair in seq.windows(2) {
                        *acc.entry((pair[0], pair[1])).or_insert(0) += 1;
                    }
                    acc
                }
            ).reduce(
                HashMap::new,
                |mut acc1, acc2| {
                    for (k, v) in acc2 {
                        *acc1.entry(k).or_insert(0) += v;
                    }
                    acc1
                }
            );

            let Some((left, right)) = Self::find_most_frequent_pair(&counts) else {
                break;
            };

            let merged_id = tokenizer.insert_merged_token(left, right);

            // Parallel replace: replace the pair in all sequences simultaneously
            sequences.par_iter_mut().for_each(|sequence| {
                Self::replace_pair_in_seq(sequence, left, right, merged_id);
            });

            if tokenizer.vocab_size() % 100 == 0 || tokenizer.vocab_size() == target_vocab_size {
                println!("Vocab size: {} / {}", tokenizer.vocab_size(), target_vocab_size);
            }
        }
        
        println!("Tokenizer training complete!");
        tokenizer
    }

    // Input: UTF-8 text.
    // Output: token IDs representing the same bytes.
    // Example: an untrained tokenizer encodes "ABC" as [65, 66, 67].
    // Algorithm: find the adjacent pair with the smallest merge rank, replace
    // all non-overlapping occurrences, and repeat until no pair is mergeable.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let b = text.as_bytes();
        let mut ids: Vec<u32> = b.iter().map(|&byte| byte as u32).collect();

        loop {
            let mut best_pair = None;
            let mut best_rank = usize::MAX;
            let mut best_merged_id = None;

            for pair in ids.windows(2) {
                let pair = (pair[0], pair[1]);

                if let Some(&rank) = self.ranks.get(&pair)
                    && rank < best_rank
                {
                    best_pair = Some(pair);
                    best_rank = rank;
                    best_merged_id = self.merges.get(&pair).copied();
                }
            }

            let Some((left, right)) = best_pair else {
                break;
            };

            let merged_id = best_merged_id.expect("merge rank has no merge ID");

            Self::replace_pair_in_seq(&mut ids, left, right, merged_id);
        }
        ids
    }

    // Input: token IDs produced by encode.
    // Output: the original UTF-8 String.
    // Example: decode([65, 66, 67]) == "ABC".
    // Edge case: decoding an empty slice should return an empty String.
    pub fn decode(&self, ids: &[u32]) -> Result<String, DecodeError> {
        let mut bytes = Vec::new();

        for &id in ids {
            let token = self
                .vocab
                .get(id as usize)
                .ok_or(DecodeError::InvalidTokenId(id))?;
            bytes.extend_from_slice(token);
        }

        String::from_utf8(bytes).map_err(DecodeError::InvalidUtf8)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    // --- Helper methods (private) ---

    fn build_initial_vocab() -> (Vec<Vec<u8>>, HashMap<Vec<u8>, u32>) {
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(1024);
        for i in 0..=255 {
            vocab.push(vec![i as u8]);
        }

        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab.len());
        for (i, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }
        (vocab, token_to_id)
    }

    // Input: [a, b, a, b] produces counts {(a,b): 2, (b,a): 1}.
    // Output: a map from (left_id, right_id) to occurrence count.
     #[allow(dead_code)]
    fn count_pair_frequencies(seq: &[u32]) -> HashMap<(u32, u32), usize> {
        let mut num_of_occur: HashMap<(u32, u32), usize> = HashMap::new();
        for j in seq.windows(2) {
            let pair = (j[0], j[1]);
            *num_of_occur.entry(pair).or_insert(0) += 1;
        }
        num_of_occur
    }

    // Input: pair-frequency map.
    // Output: Some(pair) for a non-empty map, otherwise None.
    fn find_most_frequent_pair(counts: &HashMap<(u32, u32), usize>) -> Option<(u32, u32)> {
        if counts.is_empty() {
            return None;
        }

        let mut max_value: usize = 0;
        let mut max_key = None;

        for (key, value) in counts {
            let is_higher_frequency = *value > max_value;
            let is_smaller_tie =
                *value == max_value && max_key.is_some_and(|current| *key < current);

            if is_higher_frequency || is_smaller_tie {
                max_value = *value;
                max_key = Some(*key);
            }
        }

        max_key
    }

    // Input: two valid token IDs.
    // Output: the ID of the concatenated token.
    fn insert_merged_token(&mut self, left: u32, right: u32) -> u32 {
        let mut merged: Vec<u8> = Vec::new();
        let mut l = self.vocab[left as usize].clone();
        let mut r = self.vocab[right as usize].clone();
        merged.append(&mut l);
        merged.append(&mut r);

        if self.token_to_id.contains_key(&merged) {
            return self.token_to_id[&merged];
        }

        let merged_id = self.vocab.len() as u32;

        self.vocab.push(merged.clone());
        self.token_to_id.insert(merged, merged_id);

        self.merges.insert((left, right), merged_id);
        self.ranks.insert((left, right), self.ranks.len());
        merged_id
    }

    // In-place replacement using Two-Pointer Array Mutation
    // [A, B] = Z 
    // 1. [A, B, A, B] write_idx = 0, i = 1
    // 2. [Z, B, A, B] Write_ix++, i = 3
    // 3. [Z, Z, A, B] write_idx = 1, i = 3, cant go more
    // 4. [Z, Z]
    fn replace_pair_in_seq(seq: &mut Vec<u32>, left: u32, right: u32, merged_id: u32) {
        if seq.is_empty() {
            return;
        }

        let mut i = 0;
        let mut write_idx = 0;
        while i < seq.len() {
            if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
                seq[write_idx] = merged_id;
                i += 2;
            } else {
                seq[write_idx] = seq[i];
                i += 1;
            }
            write_idx += 1;
        }
        seq.truncate(write_idx);
    }

    // Persistence
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        self.validate()?;

        let mut saved_merges = Vec::with_capacity(self.merges.len());

        for (&(left, right), &merged_id) in &self.merges {
            let rank = self.ranks[&(left, right)];

            saved_merges.push(MergeFile {
                left,
                right,
                merged_id,
                rank,
            });
        }

        saved_merges.sort_by_key(|merge| merge.rank);

        let saved_tokenizer = TokenizerFile {
            version: FILE_VERSION,
            vocab: self.vocab.clone(),
            merges: saved_merges,
        };

        let json = serde_json::to_string_pretty(&saved_tokenizer)
            .map_err(|error| Error::new(ErrorKind::InvalidData, error))?;

        fs::write(path, json)
    }

    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let saved_tokenizer: TokenizerFile = serde_json::from_str(&json)
            .map_err(|error| Error::new(ErrorKind::InvalidData, error))?;

        if saved_tokenizer.version != FILE_VERSION {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "unsupported tokenizer file version",
            ));
        }

        let mut token_to_id = HashMap::with_capacity(saved_tokenizer.vocab.len());

        for (id, token) in saved_tokenizer.vocab.iter().enumerate() {
            let id = u32::try_from(id)
                .map_err(|_| Error::new(ErrorKind::InvalidData, "the vocabulary is too large"))?;

            if token_to_id.insert(token.clone(), id).is_some() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "the vocabulary contains duplicate tokens",
                ));
            }
        }

        let mut merges = HashMap::with_capacity(saved_tokenizer.merges.len());
        let mut ranks = HashMap::with_capacity(saved_tokenizer.merges.len());

        for merge in saved_tokenizer.merges {
            let pair = (merge.left, merge.right);

            if merges.insert(pair, merge.merged_id).is_some() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "the file contains a duplicate merge pair",
                ));
            }

            ranks.insert(pair, merge.rank);
        }

        let tokenizer = Self::from_parts(saved_tokenizer.vocab, token_to_id, merges, ranks);
        tokenizer.validate()?;

        Ok(tokenizer)
    }

    fn validate(&self) -> std::io::Result<()> {
        if self.vocab.len() < 256 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the vocabulary has fewer than 256 byte tokens",
            ));
        }

        for byte in 0..=255 {
            if self.vocab[byte] != vec![byte as u8] {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "the base byte vocabulary is invalid",
                ));
            }
        }

        if self.token_to_id.len() != self.vocab.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the vocabulary maps have different sizes",
            ));
        }

        for (id, token) in self.vocab.iter().enumerate() {
            if self.token_to_id.get(token) != Some(&(id as u32)) {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "the token lookup map is invalid",
                ));
            }
        }

        if self.merges.len() != self.ranks.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "the merge and rank maps have different sizes",
            ));
        }

        let mut used_ranks = vec![false; self.ranks.len()];

        for (&(left, right), &merged_id) in &self.merges {
            let Some(&rank) = self.ranks.get(&(left, right)) else {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "a merge does not have a rank",
                ));
            };

            if rank >= used_ranks.len() || used_ranks[rank] {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "the merge ranks are not unique and continuous",
                ));
            }
            used_ranks[rank] = true;

            let Some(left_token) = self.vocab.get(left as usize) else {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "a merge has an invalid left token ID",
                ));
            };
            let Some(right_token) = self.vocab.get(right as usize) else {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "a merge has an invalid right token ID",
                ));
            };
            let Some(merged_token) = self.vocab.get(merged_id as usize) else {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "a merge has an invalid result token ID",
                ));
            };

            let mut expected_token = left_token.clone();
            expected_token.extend_from_slice(right_token);

            if *merged_token != expected_token {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "a merged token has incorrect bytes",
                ));
            }
        }

        Ok(())
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_ascii_text() {
        let tokenizer = Tokenizer::new();

        let result = tokenizer.encode("ABC");

        assert_eq!(result, vec![65, 66, 67]);
    }

    #[test]
    fn encode_empty_text() {
        let tokenizer = Tokenizer::new();

        let result = tokenizer.encode("");

        assert_eq!(result, Vec::<u32>::new());
    }

    #[test]
    fn encode_utf8_text_as_bytes() {
        let tokenizer = Tokenizer::new();

        let result = tokenizer.encode("é");

        assert_eq!(result, vec![195, 169]);
    }

    #[test]
    fn decode_ascii_tokens() {
        let tokenizer = Tokenizer::new();

        let result = tokenizer.decode(&[65, 66, 67]);

        assert_eq!(result.unwrap(), "ABC");
    }

    #[test]
    fn decode_rejects_invalid_token_id() {
        let tokenizer = Tokenizer::new();

        let result = tokenizer.decode(&[u32::MAX]);

        assert!(matches!(result, Err(DecodeError::InvalidTokenId(u32::MAX))));
    }

    #[test]
    fn decode_rejects_invalid_utf8() {
        let tokenizer = Tokenizer::from_parts(
            vec![vec![0xff, 0xfe, 0xc0, 0xaf]],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );

        let result = tokenizer.decode(&[0]);

        assert!(matches!(result, Err(DecodeError::InvalidUtf8(_))));
    }

    #[test]
    fn encode_then_decode_returns_original_tex() {
        let tokenizer = Tokenizer::new();
        let original = "héllo";

        let ids = tokenizer.encode(original);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded.unwrap(), original);
    }

    #[test]
    fn count_repeated_pairs() {
        let counts = Tokenizer::count_pair_frequencies(&[1, 2, 1, 2]);

        assert_eq!(counts.get(&(1, 2)), Some(&2));
        assert_eq!(counts.get(&(2, 1)), Some(&1));
    }

    #[test]
    fn count_overlapping_pairs() {
        let counts = Tokenizer::count_pair_frequencies(&[7, 7, 7]);

        assert_eq!(counts.get(&(7, 7)), Some(&2));
    }

    #[test]
    fn count_pairs_for_short_sequences() {
        assert!(Tokenizer::count_pair_frequencies(&[]).is_empty());
        assert!(Tokenizer::count_pair_frequencies(&[1]).is_empty());
    }

    #[test]
    fn finds_no_pair_in_empty_counts() {
        let counts = HashMap::new();

        let result = Tokenizer::find_most_frequent_pair(&counts);

        assert_eq!(result, None);
    }

    #[test]
    fn finds_pair_with_highest_frequency() {
        let mut counts = HashMap::new();
        counts.insert((10, 20), 50);
        counts.insert((20, 30), 100);
        counts.insert((30, 40), 75);

        let result = Tokenizer::find_most_frequent_pair(&counts);

        assert_eq!(result, Some((20, 30)));
    }

    #[test]
    fn finds_smallest_pair_when_frequencies_are_equal() {
        let mut counts = HashMap::new();
        counts.insert((u32::MAX, u32::MAX), usize::MAX);
        counts.insert((0, u32::MAX), usize::MAX);
        counts.insert((0, 0), usize::MAX);

        let result = Tokenizer::find_most_frequent_pair(&counts);

        assert_eq!(result, Some((0, 0)));
    }

    #[test]
    fn replaces_pair_at_beginning() {
        let mut vec = vec![10, 20, 30];
        Tokenizer::replace_pair_in_seq(&mut vec, 10, 20, 100);

        assert_eq!(vec, vec![100, 30]);
    }

    #[test]
    fn replaces_pair_in_middle() {
        let mut vec = vec![10, 20, 30, 40];
        let result = Tokenizer::replace_pair_in_seq(&mut vec, 20, 30, 100);

        assert_eq!(vec, vec![10, 100, 40]);
    }

    #[test]
    fn replaces_pair_at_end() {
        let mut vec = vec![10, 20, 30];
        let result = Tokenizer::replace_pair_in_seq(&mut vec, 20, 30, 100);

        assert_eq!(vec, vec![10, 100]);
    }

    #[test]
    fn keeps_sequence_when_pair_does_not_match() {
        let mut vec = vec![10, 20, 30];
        let result = Tokenizer::replace_pair_in_seq(&mut vec, 40, 50, 100);

        assert_eq!(vec, vec![10, 20, 30]);
    }

    #[test]
    fn replaces_overlapping_pair_from_left_to_right() {
        let mut vec = vec![10, 10, 10];
        let result = Tokenizer::replace_pair_in_seq(&mut vec, 10, 10, 100);

        assert_eq!(vec, vec![100, 10]);
    }

    #[test]
    fn replaces_multiple_sequential_pairs() {
        let mut vec = vec![10, 20, 10, 20];
        Tokenizer::replace_pair_in_seq(&mut vec, 10, 20, 100);
        assert_eq!(vec, vec![100, 100]);
    }

    #[test]
    fn handles_sequence_of_length_one() {
        let mut vec = vec![10];
        Tokenizer::replace_pair_in_seq(&mut vec, 10, 20, 100);
        assert_eq!(vec, vec![10]);
    }

    #[test]
    fn handles_empty_sequence() {
        let mut vec: Vec<u32> = vec![];
        Tokenizer::replace_pair_in_seq(&mut vec, 10, 20, 100);
        assert_eq!(vec, vec![]);
    }

    #[test]
    fn inserts_new_merged_token() {
        let mut tokenizer = Tokenizer::new();

        let merged_id = tokenizer.insert_merged_token(97, 98);

        assert_eq!(merged_id, 256);
        assert_eq!(tokenizer.vocab[256], vec![b'a', b'b']);
        assert_eq!(tokenizer.token_to_id.get(&vec![b'a', b'b']), Some(&256));
        assert_eq!(tokenizer.merges.get(&(97, 98)), Some(&256));
        assert_eq!(tokenizer.ranks.get(&(97, 98)), Some(&0));
    }

    #[test]
    fn trains_first_merge() {
        let tokenizer = Tokenizer::train(["aaaa"], 257);

        assert_eq!(tokenizer.vocab_size(), 257);
        assert_eq!(tokenizer.vocab[256], vec![b'a', b'a']);
        assert_eq!(tokenizer.token_to_id.get(&vec![b'a', b'a']), Some(&256));
        assert_eq!(tokenizer.merges.get(&(97, 97)), Some(&256));
        assert_eq!(tokenizer.ranks.get(&(97, 97)), Some(&0));
    }

    #[test]
    fn training_keeps_texts_separate() {
        let tokenizer = Tokenizer::train(["ab", "cd"], 258);

        assert_eq!(tokenizer.vocab_size(), 258);
        assert!(tokenizer.merges.contains_key(&(97, 98)));
        assert!(!tokenizer.merges.contains_key(&(98, 99)));
    }

    #[test]
    fn target_below_base_vocabulary_keeps_all_byte_tokens() {
        let tokenizer = Tokenizer::train(["aaaa"], 100);

        assert_eq!(tokenizer.vocab_size(), 256);
    }

    #[test]
    fn training_stops_when_no_pairs_remain() {
        let tokenizer = Tokenizer::train(["a"], 300);

        assert_eq!(tokenizer.vocab_size(), 256);
        assert!(tokenizer.merges.is_empty());
    }

    #[test]
    fn encodes_with_learned_merge() {
        let tokenizer = Tokenizer::train(["aaaa"], 257);

        assert_eq!(tokenizer.encode("aaaa"), vec![256, 256]);
        assert_eq!(tokenizer.decode(&[256, 256]).unwrap(), "aaaa");
    }

    #[test]
    fn encodes_with_learned_merge2() {
        let tokenizer = Tokenizer::train(["aaaaaaaa"], 258);

        assert_eq!(tokenizer.encode("aaaaaaaa"), vec![257, 257]);
        assert_eq!(tokenizer.decode(&[257, 257]).unwrap(), "aaaaaaaa");
        assert_eq!(tokenizer.encode("aaaa"), vec![257]);
    }

    #[test]
    fn trained_encode_then_decode_returns_original_text() {
        let tokenizer = Tokenizer::train(["aaaaaaaa"], 258);
        let text = "aaaaaaaa";

        assert_eq!(tokenizer.decode(&tokenizer.encode(text)).unwrap(), text);
    }

    #[test]
    fn saves_and_loads_trained_tokenizer() {
        let path = std::env::temp_dir().join(format!(
            "bpe_tokenizer_round_trip_{}.json",
            std::process::id()
        ));
        let tokenizer = Tokenizer::train(["banana banana", "bandana"], 270);

        tokenizer.save_to_file(&path).unwrap();
        let loaded = Tokenizer::load_from_file(&path).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(loaded.vocab, tokenizer.vocab);
        assert_eq!(loaded.token_to_id, tokenizer.token_to_id);
        assert_eq!(loaded.merges, tokenizer.merges);
        assert_eq!(loaded.ranks, tokenizer.ranks);
        assert_eq!(
            loaded.encode("banana bandana"),
            tokenizer.encode("banana bandana")
        );
    }

    #[test]
    fn load_rejects_invalid_json() {
        let path =
            std::env::temp_dir().join(format!("bpe_tokenizer_invalid_{}.json", std::process::id()));
        std::fs::write(&path, "this is not valid JSON").unwrap();

        let result = Tokenizer::load_from_file(&path);
        std::fs::remove_file(path).unwrap();

        assert!(matches!(result, Err(error) if error.kind() == ErrorKind::InvalidData));
    }

    #[test]
    fn load_rejects_unsupported_file_version() {
        let path =
            std::env::temp_dir().join(format!("bpe_tokenizer_version_{}.json", std::process::id()));
        let tokenizer = Tokenizer::new();
        tokenizer.save_to_file(&path).unwrap();

        let json = std::fs::read_to_string(&path).unwrap();
        let mut saved_tokenizer: TokenizerFile = serde_json::from_str(&json).unwrap();
        saved_tokenizer.version = FILE_VERSION + 1;
        let json = serde_json::to_string_pretty(&saved_tokenizer).unwrap();
        std::fs::write(&path, json).unwrap();

        let result = Tokenizer::load_from_file(&path);
        std::fs::remove_file(path).unwrap();

        assert!(matches!(result, Err(error) if error.kind() == ErrorKind::InvalidData));
    }

    #[test]
    fn load_rejects_invalid_merge_reference() {
        let path =
            std::env::temp_dir().join(format!("bpe_tokenizer_merge_{}.json", std::process::id()));
        let tokenizer = Tokenizer::train(["aaaa"], 257);
        tokenizer.save_to_file(&path).unwrap();

        let json = std::fs::read_to_string(&path).unwrap();
        let mut saved_tokenizer: TokenizerFile = serde_json::from_str(&json).unwrap();
        saved_tokenizer.merges[0].left = u32::MAX;
        let json = serde_json::to_string_pretty(&saved_tokenizer).unwrap();
        std::fs::write(&path, json).unwrap();

        let result = Tokenizer::load_from_file(&path);
        std::fs::remove_file(path).unwrap();

        assert!(matches!(result, Err(error) if error.kind() == ErrorKind::InvalidData));
    }
}
