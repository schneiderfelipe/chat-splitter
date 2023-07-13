# `chat-splitter`

`chat-splitter` is a utility for managing chat history in applications using the [64bit/async-openai](https://github.com/64bit/async-openai) and [zurawiki/tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) libraries.
It provides functionality for splitting chats into manageable chunks,
ensuring that the maximum number of tokens and messages are not exceeded.

## Usage

To use `chat-splitter`,
first create a `ChatSplitter` with the desired model,
maximum tokens,
and maximum messages.
Then,
use the `split` method to split your chat history into two groups: the 'outdated' messages and the 'recent' ones.

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
