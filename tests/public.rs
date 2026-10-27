//! 公开测试：cargo test 全绿是交作业的最低要求

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bpe-test-{}-{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}
fn bpe(args: &[&str], d: &PathBuf) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mini-bpe")).args(args).current_dir(d).output().unwrap()
}
fn ok(args: &[&str], d: &PathBuf) -> String {
    let o = bpe(args, d);
    assert!(o.status.success(), "{:?} 失败：{}", args, String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).trim_end().to_string()
}

const CORPUS: &str = "Rust 是一门系统编程语言。Rust 的所有权系统让内存安全不需要垃圾回收。\
借用检查器在编译期检查引用。trait 让不同类型共享行为。迭代器是零成本抽象。\
所有权、借用、生命周期是 Rust 最难的三个概念。🦀 是 Rust 的吉祥物。";

fn trained(name: &str, vocab: usize) -> PathBuf {
    let d = dir(name);
    fs::write(d.join("corpus.txt"), CORPUS.repeat(3)).unwrap();
    ok(&["train", "corpus.txt", "--vocab-size", &vocab.to_string(), "-o", "m.json"], &d);
    d
}

#[test]
fn wikipedia_example_exact_ids() {
    let d = dir("wiki");
    fs::write(d.join("c.txt"), "aaabdaaabac").unwrap();
    ok(&["train", "c.txt", "--vocab-size", "259", "-o", "m.json"], &d);
    assert_eq!(ok(&["encode", "-m", "m.json", "aaabdaaabac"], &d), "258 100 258 97 99");
}

#[test]
fn tie_break_smallest_pair() {
    // 第一轮 aa=4 次；第二轮 (256,97)=2 和 (97,98)=2 并列 → 取小的 (97,98)
    let d = dir("tie");
    fs::write(d.join("c.txt"), "aaabdaaabac").unwrap();
    ok(&["train", "c.txt", "--vocab-size", "258", "-o", "m.json"], &d);
    let m = fs::read_to_string(d.join("m.json")).unwrap();
    assert!(m.replace(' ', "").contains("[[97,97],[97,98]]"), "合并顺序不对：{}", m);
}

#[test]
fn roundtrip_chinese_and_emoji() {
    let d = trained("rt", 400);
    for s in ["Rust 的所有权", "🦀🦀 借用检查器", "完全没见过的句子：并发与异步", "", "a"] {
        let ids = ok(&["encode", "-m", "m.json", s], &d);
        if s.is_empty() {
            assert_eq!(ids, "");
            continue;
        }
        let args: Vec<&str> = ["decode", "-m", "m.json"].into_iter().chain(ids.split(' ')).collect();
        assert_eq!(ok(&args, &d), s);
    }
}

#[test]
fn training_compresses() {
    let d = trained("compress", 400);
    let n = ok(&["encode", "-m", "m.json", "Rust 的所有权系统"], &d).split(' ').count();
    let bytes = "Rust 的所有权系统".len();
    assert!(n * 2 < bytes, "训练过的词应该被压缩：{} 字节 → {} 个 token", bytes, n);
}

#[test]
fn bigger_vocab_never_longer() {
    let small = trained("small", 300);
    let big = trained("big", 450);
    let s = "所有权、借用、生命周期";
    let a = ok(&["encode", "-m", "m.json", s], &small).split(' ').count();
    let b = ok(&["encode", "-m", "m.json", s], &big).split(' ').count();
    assert!(b <= a);
}

#[test]
fn stops_early_when_nothing_repeats() {
    let d = dir("early");
    fs::write(d.join("c.txt"), "abcdef").unwrap();
    ok(&["train", "c.txt", "--vocab-size", "1000", "-o", "m.json"], &d);
    assert_eq!(ok(&["encode", "-m", "m.json", "abcdef"], &d), "97 98 99 100 101 102");
}

#[test]
fn errors() {
    let d = trained("err", 300);
    assert_eq!(bpe(&["train", "corpus.txt", "--vocab-size", "100", "-o", "x.json"], &d).status.code(), Some(1));
    assert_eq!(bpe(&["decode", "-m", "m.json", "99999"], &d).status.code(), Some(1));
    assert_eq!(bpe(&["encode", "-m", "nope.json", "hi"], &d).status.code(), Some(1));
    fs::write(d.join("bad.json"), "{\"merges\": [[300, 1]]}").unwrap();
    assert_eq!(bpe(&["encode", "-m", "bad.json", "hi"], &d).status.code(), Some(1), "引用不存在 id 的模型要拒绝");
}
