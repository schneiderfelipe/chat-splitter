# `chat-splitter`

Split chats into 'outdated' and 'recent' messages,
so that you never exceed OpenAI's maximum number of tokens when using [64bit/async-openai](https://github.com/64bit/async-openai).
Tokens are counted using [zurawiki/tiktoken-rs](https://github.com/zurawiki/tiktoken-rs).

You can split by
both
maximum message count and
maximum chat completion token count.

## Usage

Here's a basic example:

```rust
// Get all your previously stored chat messages...
let mut stored_messages = /* get_stored_messages()? */;

// ...and split into 'outdated' and 'recent',
// where 'recent' always fits the model's context window.
let (outdated_messages, recent_messages) = ChatSplitter::default()
    .split(&stored_messages);
```

For a more detailed example,
see [`examples/chat.rs`](https://github.com/schneiderfelipe/chat-splitter/blob/main/examples/chat.rs).

## Contributing

Contributions to `chat-splitter` are welcome!
If you find a bug or have a feature request,
please [submit an issue](https://github.com/schneiderfelipe/chat-splitter/issues).
If you'd like to contribute code,
please feel free to [submit a pull request](https://github.com/schneiderfelipe/chat-splitter/pulls).
