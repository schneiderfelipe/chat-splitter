# chat-memory

Chat memory is a utility for managing chat history in applications using the [64bit/async-openai](https://github.com/64bit/async-openai) library. It provides functionality for splitting chats into manageable chunks, ensuring that the maximum number of tokens and messages are not exceeded.

## Usage

To use chat-memory, first create a `ChatSplitter` with the desired model, maximum tokens, and maximum messages. Then, use the `split` method to split your chat history into two groups: the old messages and the most recent ones.

Here's a basic example:

```rust
let mut stored_messages = get_stored_messages()?;
stored_messages.push(
    ChatCompletionRequestMessageArgs::default()
        .role(Role::User)
        .content("Where was it played?")
        .build()?,
);
assert!(stored_messages.len() > MAX_MESSAGES);

let (_previous_messages, recent_messages) = ChatSplitter::new(MODEL)
    .max_tokens(MAX_TOKENS)
    .max_messages(MAX_MESSAGES)
    .split(&stored_messages);
```

For a more detailed example, see `examples/chat.rs`.

## Contributing

Contributions to chat-memory are welcome! If you find a bug or have a feature request, please [submit an issue](https://github.com/schneiderfelipe/chat-memory/issues). If you'd like to contribute code, please feel free to [submit a pull request](https://github.com/schneiderfelipe/chat-memory/pulls).
