use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_DIR_ID: AtomicUsize = AtomicUsize::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let id = NEXT_DIR_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mini-bpe-mytest-{label}-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temporary test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[&str], dir: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mini-bpe"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run mini-bpe")
}

fn run_with_stdin(args: &[&str], dir: &Path, input: &[u8]) -> Output {
    let mut child: Child = Command::new(env!("CARGO_BIN_EXE_mini-bpe"))
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start mini-bpe with piped stdin");

    child
        .stdin
        .take()
        .expect("child stdin is piped")
        .write_all(input)
        .expect("write child stdin");
    child.wait_with_output().expect("wait for mini-bpe")
}

fn write_model(dir: &Path, name: &str, json: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, json).expect("write model fixture");
    path
}

fn assert_cli_error(output: &Output) {
    assert_eq!(output.status.code(), Some(1), "unexpected exit status");
    assert!(output.stdout.is_empty(), "error should not write to stdout");
    assert!(
        !output.stderr.is_empty(),
        "error should explain itself on stderr"
    );
}

#[test]
fn train_handles_empty_minimum_overlapping_and_huge_vocab_limits() {
    let dir = TestDir::new("train-boundaries");

    fs::write(dir.path().join("empty.txt"), "").unwrap();
    let output = run(
        &[
            "train",
            "empty.txt",
            "--vocab-size",
            "256",
            "-o",
            "empty.json",
        ],
        dir.path(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert_eq!(
        fs::read_to_string(dir.path().join("empty.json")).unwrap(),
        "{\"merges\":[]}"
    );

    fs::write(dir.path().join("overlap.txt"), "aaaaa").unwrap();
    let output = run(
        &[
            "train",
            "overlap.txt",
            "--vocab-size",
            "257",
            "-o",
            "overlap.json",
        ],
        dir.path(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("overlap.json")).unwrap(),
        "{\"merges\":[[97,97]]}"
    );
    let output = run(&["encode", "-m", "overlap.json", "aaaaa"], dir.path());
    assert!(output.status.success());
    assert_eq!(output.stdout, b"256 256 97\n");

    let output = run(
        &[
            "train",
            "empty.txt",
            "--vocab-size",
            &usize::MAX.to_string(),
            "-o",
            "huge.json",
        ],
        dir.path(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert_eq!(
        fs::read_to_string(dir.path().join("huge.json")).unwrap(),
        "{\"merges\":[]}"
    );
}

#[test]
fn encode_uses_rule_rank_and_merges_all_non_overlapping_matches() {
    let dir = TestDir::new("encode-priority");
    write_model(dir.path(), "rank.json", "{\"merges\":[[98,99],[97,98]]}");

    // The right pair (b,c) has rank 0, so it wins over the left pair (a,b).
    let output = run(&["encode", "-m", "rank.json", "abc"], dir.path());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"97 256\n");
    let output = run(&["decode", "-m", "rank.json", "97", "256"], dir.path());
    assert!(output.status.success());
    assert_eq!(output.stdout, b"abc");

    write_model(dir.path(), "same-pair.json", "{\"merges\":[[97,98]]}");
    let output = run(&["encode", "-m", "same-pair.json", "ababab"], dir.path());
    assert!(output.status.success());
    assert_eq!(output.stdout, b"256 256 256\n");
}

#[test]
fn encode_and_decode_preserve_unicode_nul_and_line_endings() {
    let dir = TestDir::new("utf8-bytes");
    write_model(dir.path(), "empty.json", "{\"merges\":[]}");
    let input = "é中🦀\r\n\0".as_bytes();

    let encoded = run_with_stdin(&["encode", "-m", "empty.json"], dir.path(), input);
    assert!(
        encoded.status.success(),
        "{}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    assert!(encoded.stderr.is_empty());
    let ids = String::from_utf8(encoded.stdout).unwrap();
    let id_args: Vec<&str> = ids.trim().split(' ').collect();
    let mut decode_args = vec!["decode", "-m", "empty.json"];
    decode_args.extend(id_args);
    let decoded = run(&decode_args, dir.path());
    assert!(
        decoded.status.success(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    assert_eq!(decoded.stdout, input);

    // The three UTF-8 bytes of 中 may arrive as three separate base tokens.
    let decoded = run(
        &["decode", "-m", "empty.json", "228", "184", "173"],
        dir.path(),
    );
    assert!(decoded.status.success());
    assert_eq!(decoded.stdout, "中".as_bytes());
}

#[test]
fn invalid_utf8_corpus_stdin_and_decode_bytes_fail_cleanly() {
    let dir = TestDir::new("invalid-utf8");
    fs::write(dir.path().join("bad.bin"), [b'a', 0xff, b'b']).unwrap();
    let output = run(
        &[
            "train",
            "bad.bin",
            "--vocab-size",
            "256",
            "-o",
            "should-not-exist.json",
        ],
        dir.path(),
    );
    assert_cli_error(&output);
    assert!(String::from_utf8_lossy(&output.stderr).contains("读取语料"));
    assert!(!dir.path().join("should-not-exist.json").exists());

    write_model(dir.path(), "empty.json", "{\"merges\":[]}");
    let output = run_with_stdin(&["encode", "-m", "empty.json"], dir.path(), &[0xff]);
    assert_cli_error(&output);

    for ids in [
        vec!["195", "40"],
        vec!["255"],
        vec!["192", "175"],
        vec!["237", "160", "128"],
    ] {
        let mut args = vec!["decode", "-m", "empty.json"];
        args.extend(ids);
        let output = run(&args, dir.path());
        assert_cli_error(&output);
        assert!(String::from_utf8_lossy(&output.stderr).contains("不是合法 UTF-8"));
    }
}

#[test]
fn malformed_models_and_unknown_ids_are_rejected() {
    let dir = TestDir::new("invalid-models");
    let invalid_models = [
        "{}",
        "{\"merges\":null}",
        "[]",
        "{\"merges\":[[97]]}",
        "{\"merges\":[[-1,97]]}",
        "{\"merges\":[[4294967296,97]]}",
        "{\"merges\":[[256,97]]}",
        "{\"merges\":[[97,256],[97,97]]}",
        "{\"merges\":",
    ];

    for (index, json) in invalid_models.iter().enumerate() {
        let name = format!("invalid-{index}.json");
        write_model(dir.path(), &name, json);
        let output = run(&["encode", "-m", &name, "hello"], dir.path());
        assert_cli_error(&output);
    }

    write_model(dir.path(), "empty.json", "{\"merges\":[]}");
    for id in ["-1", "4294967295", "4294967296"] {
        let output = run(&["decode", "-m", "empty.json", id], dir.path());
        assert_cli_error(&output);
    }
}

#[test]
fn vocab_top_is_clamped_and_displays_raw_bytes() {
    let dir = TestDir::new("vocab-top");
    write_model(dir.path(), "partial.json", "{\"merges\":[[228,184]]}");

    let output = run(&["vocab", "-m", "partial.json", "--top", "1"], dir.path());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, "256\t\"�\"\t[228, 184]\n".as_bytes());

    let output = run(&["vocab", "-m", "partial.json", "--top", "0"], dir.path());
    assert!(output.status.success());
    assert!(output.stdout.is_empty());

    let output = run(
        &[
            "vocab",
            "-m",
            "partial.json",
            "--top",
            &usize::MAX.to_string(),
        ],
        dir.path(),
    );
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("256\t"));

    let output = run(&["vocab", "--help"], dir.path());
    assert!(output.status.success());
    assert!(output.stdout.windows(6).any(|window| window == b"--top "));
    assert!(output.stderr.is_empty());
}

#[test]
fn training_is_deterministic_and_can_overwrite_its_input() {
    let dir = TestDir::new("deterministic");
    fs::write(dir.path().join("corpus.txt"), "aaabdaaabac".repeat(5)).unwrap();

    let mut first_model = None;
    for index in 0..5 {
        let model_name = format!("model-{index}.json");
        let output = run(
            &[
                "train",
                "corpus.txt",
                "--vocab-size",
                "275",
                "-o",
                &model_name,
            ],
            dir.path(),
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let model = fs::read(dir.path().join(model_name)).unwrap();
        if let Some(expected) = &first_model {
            assert_eq!(&model, expected, "training result must be deterministic");
        } else {
            first_model = Some(model);
        }
    }

    fs::write(dir.path().join("same-file.txt"), "xyxyxy").unwrap();
    let output = run(
        &[
            "train",
            "same-file.txt",
            "--vocab-size",
            "257",
            "-o",
            "same-file.txt",
        ],
        dir.path(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("same-file.txt")).unwrap(),
        "{\"merges\":[[120,121]]}"
    );
}

#[test]
fn deep_merge_chain_decodes_without_recursive_stack_growth() {
    let dir = TestDir::new("deep-chain");
    let merge_count: u32 = 12_000;
    let mut json = String::from("{\"merges\":[[97,97]");
    for rank in 1..merge_count {
        json.push_str(&format!(",[{},97]", 256 + rank - 1));
    }
    json.push_str("]}");
    fs::write(dir.path().join("deep.json"), json).unwrap();

    let final_id = 256 + merge_count - 1;
    let output = run(
        &["decode", "-m", "deep.json", &final_id.to_string()],
        dir.path(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, vec![b'a'; merge_count as usize + 1]);
}

#[test]
fn paths_with_spaces_and_hyphen_leading_text_work() {
    let dir = TestDir::new("paths-and-args");
    let nested = dir.path().join("folder with spaces");
    fs::create_dir_all(&nested).unwrap();
    fs::write(nested.join("corpus file.txt"), "aaaa").unwrap();
    let corpus = nested
        .join("corpus file.txt")
        .to_string_lossy()
        .into_owned();
    let model = nested
        .join("model file.json")
        .to_string_lossy()
        .into_owned();
    let output = run(
        &["train", &corpus, "--vocab-size", "257", "-o", &model],
        dir.path(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = run(&["encode", "-m", &model, "-hello"], dir.path());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"45 104 101 108 108 111\n");
}

