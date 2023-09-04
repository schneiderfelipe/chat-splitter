# chat-splitter

[![Build Status]][actions]
[![Latest Version]][crates.io]
[![Documentation]][docs.rs]

[Build Status]: https://github.com/schneiderfelipe/chat-splitter/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/schneiderfelipe/chat-splitter/actions/workflows/rust.yml
[Latest Version]: https://img.shields.io/crates/v/chat_splitter.svg
[crates.io]: https://crates.io/crates/chat_splitter
[Documentation]: https://img.shields.io/docsrs/chat-splitter
[docs.rs]: https://docs.rs/chat-splitter

> For more information,
> please refer to the [blog announcement](https://schneiderfelipe.github.io/posts/chat-splitter-first-release/).

When utilizing the [`async_openai`](https://github.com/64bit/async-openai) [Rust](https://www.rust-lang.org/) crate,
it is crucial to ensure that you do not exceed
the [maximum number of tokens](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) specified by [OpenAI](https://openai.com/)'s [chat models](https://platform.openai.com/docs/api-reference/chat).

[`chat-splitter`](https://crates.io/crates/chat_splitter) categorizes chat messages into 'outdated' and 'recent' messages,
allowing you to split them based on both the maximum
message count and the maximum chat completion token count.
The token counting functionality is provided by
[`tiktoken_rs`](https://github.com/zurawiki/tiktoken-rs).

## Usage

Here's a basic example:

```rust
// Get all your previously stored chat messages...
let mut stored_messages = /* get_stored_messages()? */;

// ...and split into 'outdated' and 'recent',
// where 'recent' always fits the context size.
let (outdated_messages, recent_messages) =
    ChatSplitter::default().split(&stored_messages);
```

For a more detailed example,
see [`examples/chat.rs`](https://github.com/schneiderfelipe/chat-splitter/blob/main/examples/chat.rs).

## Contributing

Contributions to `chat-splitter` are welcome!
If you find a bug or have a feature request,
please [submit an issue](https://github.com/schneiderfelipe/chat-splitter/issues).
If you'd like to contribute code,
please feel free to [submit a pull request](https://github.com/schneiderfelipe/chat-splitter/pulls).

License: MIT
