# chat-splitter

Never exceed [OpenAI](https://openai.com/)'s [chat models](https://platform.openai.com/docs/api-reference/chat)' [maximum number of tokens](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) when using the [`async_openai`](https://github.com/64bit/async-openai) Rust crate.

`chat-splitter` splits chats into 'outdated' and 'recent' messages.
You can split by
both
maximum message count and
maximum chat completion token count.
We use [`tiktoken_rs`](https://github.com/zurawiki/tiktoken-rs) for counting tokens.

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
