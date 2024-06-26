//! [![Build Status]][actions] [![Latest Version]][crates.io]
//!
//! [Build Status]: https://github.com/schneiderfelipe/chat-splitter/actions/workflows/rust.yml/badge.svg
//! [actions]: https://github.com/schneiderfelipe/chat-splitter/actions/workflows/rust.yml
//! [Latest Version]: https://img.shields.io/crates/v/chat_splitter.svg
//! [crates.io]: https://crates.io/crates/chat_splitter
//!
//! > For more information,
//! > please refer to the [blog announcement](https://schneiderfelipe.github.io/posts/chat-splitter-first-release/).
//!
//! When utilizing the [`async_openai`](https://github.com/64bit/async-openai) [Rust](https://www.rust-lang.org/) crate,
//! it is crucial to ensure that you do not exceed
//! the [maximum number of tokens](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) specified by [OpenAI](https://openai.com/)'s [chat models](https://platform.openai.com/docs/api-reference/chat).
//!
//! [`chat-splitter`](https://crates.io/crates/chat_splitter) categorizes chat messages into 'outdated' and 'recent' messages,
//! allowing you to split them based on both the maximum
//! message count and the maximum chat completion token count.
//! The token counting functionality is provided by
//! [`tiktoken_rs`](https://github.com/zurawiki/tiktoken-rs).
//!
//! # Usage
//!
//! Here's a basic example:
//!
//! ```ignore
//! // Get all your previously stored chat messages...
//! let mut stored_messages = /* get_stored_messages()? */;
//!
//! // ...and split into 'outdated' and 'recent',
//! // where 'recent' always fits the context size.
//! let (outdated_messages, recent_messages) =
//!     ChatSplitter::default().split(&stored_messages);
//! ```
//!
//! For a more detailed example,
//! see [`examples/chat.rs`](https://github.com/schneiderfelipe/chat-splitter/blob/main/examples/chat.rs).
//!
//! # Contributing
//!
//! Contributions to `chat-splitter` are welcome!
//! If you find a bug or have a feature request,
//! please [submit an issue](https://github.com/schneiderfelipe/chat-splitter/issues).
//! If you'd like to contribute code,
//! please feel free to [submit a pull request](https://github.com/schneiderfelipe/chat-splitter/pulls).

use std::cmp::Ordering;

use indxvec::Search;
use tiktoken_rs::get_chat_completion_max_tokens;
use tiktoken_rs::model::get_context_size;

/// Chat splitter for [OpenAI](https://openai.com/)'s [chat models](https://platform.openai.com/docs/api-reference/chat) when using [`async_openai`].
///
/// For more detailed information,
/// see the [crate documentation](`crate`).
#[derive(Clone, Debug)]
pub struct ChatSplitter {
    /// The model to use for tokenization,
    /// e.g.,
    /// `gpt-3.5-turbo`.
    ///
    /// It is passed to [`tiktoken_rs`] to select the correct tokenizer.
    model: String,

    /// The maximum number of tokens to leave for chat completion.
    ///
    /// This is the same as in the [official API](https://platform.openai.com/docs/api-reference/chat#completions/create-prompt) and given to [`async_openai`].
    /// The total length of input tokens and generated tokens is limited by the
    /// model's context size.
    /// Splits will have at least that many tokens
    /// available for chat completion,
    /// never less.
    max_tokens: u16,

    /// The maximum number of messages to have in the chat.
    ///
    /// Splits will have at most that many messages,
    /// never more.
    max_messages: usize,
}

/// Hard limit that seems to be imposed by the `OpenAI` API.
const MAX_MESSAGES_LIMIT: usize = 2_048;

/// Recommended minimum for maximum chat completion tokens.
const RECOMMENDED_MIN_MAX_TOKENS: u16 = 256;

impl Default for ChatSplitter {
    #[inline]
    fn default() -> Self {
        Self::new("gpt-3.5-turbo")
    }
}

impl ChatSplitter {
    /// Create a new [`ChatSplitter`] for the given model.
    ///
    /// # Panics
    ///
    /// If for some reason [`tiktoken_rs`] gives a context size twice as large
    /// as what would fit in a [`u16`].
    /// If this happens,
    /// it should be considered a bug,
    /// but this behaviour might change in the future,
    /// as models with larger context sizes are released.
    #[inline]
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        let max_tokens = u16::try_from(get_context_size(&model) / 2).unwrap();

        let max_messages = MAX_MESSAGES_LIMIT / 2;

        Self {
            model,
            max_tokens,
            max_messages,
        }
    }

    /// Set the maximum number of messages to have in the chat.
    ///
    /// Splits will have at most that many messages,
    /// never more.
    #[inline]
    #[must_use]
    pub fn max_messages(mut self, max_messages: impl Into<usize>) -> Self {
        self.max_messages = max_messages.into();
        if self.max_messages > MAX_MESSAGES_LIMIT {
            log::warn!(
                "max_messages = {} > {MAX_MESSAGES_LIMIT}",
                self.max_messages
            );
        }
        self
    }

    /// Set the maximum number of tokens to leave for chat completion.
    ///
    /// This is the same as in the [official API](https://platform.openai.com/docs/api-reference/chat#completions/create-prompt) and given to [`async_openai`].
    /// The total length of input tokens and generated tokens is limited by the
    /// model's context size.
    /// Splits will have at least that many tokens
    /// available for chat completion,
    /// never less.
    #[inline]
    #[must_use]
    pub fn max_tokens(mut self, max_tokens: impl Into<u16>) -> Self {
        self.max_tokens = max_tokens.into();
        if self.max_tokens < RECOMMENDED_MIN_MAX_TOKENS {
            log::warn!(
                "max_tokens = {} < {RECOMMENDED_MIN_MAX_TOKENS}",
                self.max_tokens
            );
        }
        self
    }

    /// Set the model to use for tokenization,
    /// e.g.,
    /// `gpt-3.5-turbo`.
    ///
    /// It is passed to [`tiktoken_rs`] to select the correct tokenizer.
    #[inline]
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Get a split position by only considering `max_messages`.
    #[inline]
    fn position_by_max_messages<M>(&self, messages: &[M]) -> usize {
        let upper_limit = self.max_messages.min(MAX_MESSAGES_LIMIT);

        let n = messages.len();
        let n = if n <= upper_limit { 0 } else { n - upper_limit };
        debug_assert!(messages[n..].len() <= upper_limit);
        n
    }

    /// Get a split position by only considering `max_tokens`.
    ///
    /// # Panics
    ///
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    fn position_by_max_tokens<M>(&self, messages: &[M]) -> usize
    where
        M: IntoChatCompletionRequestMessage + Clone,
    {
        let max_tokens = self.max_tokens as usize;
        let lower_limit = max_tokens.min(get_context_size(&self.model));

        let messages: Vec<_> = messages
            .iter()
            .cloned()
            .map(IntoChatCompletionRequestMessage::into_tiktoken_rs)
            .collect();

        let (n, _range) = (0..=messages.len()).binary_any(|n| {
            debug_assert!(n < messages.len());

            let tokens = get_chat_completion_max_tokens(&self.model, &messages[n..])
                .expect("tokenizer should be available");

            let cmp = tokens.cmp(&lower_limit);
            debug_assert_ne!(cmp, Ordering::Equal);
            cmp
        });

        debug_assert!(
            get_chat_completion_max_tokens(&self.model, &messages[n..])
                .expect("tokenizer should be available")
                >= lower_limit
        );
        n
    }

    /// Get a split position by first considering the `max_messages` limit,
    /// then
    /// the `max_tokens` limit.
    ///
    /// # Panics
    ///
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    fn position<M>(&self, messages: &[M]) -> usize
    where
        M: IntoChatCompletionRequestMessage + Clone,
    {
        let n = self.position_by_max_messages(messages);
        n + self.position_by_max_tokens(&messages[n..])
    }

    /// Split the chat into two groups of messages,
    /// the 'outdated' and the
    /// 'recent' ones.
    ///
    /// The 'recent' messages are guaranteed to satisfy the given limits,
    /// while
    /// the 'outdated' ones contain all the ones before 'recent'.
    ///
    /// For a detailed usage example,
    /// see [`examples/chat.rs`](https://github.com/schneiderfelipe/chat-splitter/blob/main/examples/chat.rs).
    ///
    /// # Panics
    ///
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    pub fn split<'a, M>(&self, messages: &'a [M]) -> (&'a [M], &'a [M])
    where
        M: IntoChatCompletionRequestMessage + Clone,
    {
        messages.split_at(self.position(messages))
    }
}

/// Extension trait for converting between different chat completion request
/// message types.
///
/// For a usage example,
/// see [`examples/chat.rs`](https://github.com/schneiderfelipe/chat-splitter/blob/736f4fceb57bc12adb2b70deb990030a266a95a5/examples/chat.rs#L44-L55).
pub trait IntoChatCompletionRequestMessage {
    /// Convert to [`tiktoken_rs` chat completion request message
    /// type](`tiktoken_rs::ChatCompletionRequestMessage`).
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage;

    /// Convert to [`async_openai` chat completion request message
    /// type](`async_openai::types::ChatCompletionRequestMessage`).
    fn into_async_openai(self) -> async_openai::types::ChatCompletionRequestMessage;
}

impl IntoChatCompletionRequestMessage for tiktoken_rs::ChatCompletionRequestMessage {
    #[inline]
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage {
        self
    }

    #[inline]
    fn into_async_openai(self) -> async_openai::types::ChatCompletionRequestMessage {
        match self.role.as_ref() {
            "user" => async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    role: match self.role.as_ref() {
                        "user" => async_openai::types::Role::User,
                        "system" => async_openai::types::Role::System,
                        "assistant" => async_openai::types::Role::Assistant,
                        "function" => async_openai::types::Role::Function,
                        role => panic!("unsupported role '{role}'"),
                    },
                    content: match self.content {
                        Some(text) => {
                            async_openai::types::ChatCompletionRequestUserMessageContent::Text(text)
                        }
                        None => {
                            async_openai::types::ChatCompletionRequestUserMessageContent::Array(
                                Vec::new(),
                            )
                        }
                    },
                    name: self.name,
                },
            ),
            "system" => async_openai::types::ChatCompletionRequestMessage::System(
                async_openai::types::ChatCompletionRequestSystemMessage {
                    role: match self.role.as_ref() {
                        "user" => async_openai::types::Role::User,
                        "system" => async_openai::types::Role::System,
                        "assistant" => async_openai::types::Role::Assistant,
                        "function" => async_openai::types::Role::Function,
                        role => panic!("unsupported role '{role}'"),
                    },
                    content: self
                        .content
                        .expect("system message content should be valid"),
                    name: self.name,
                },
            ),
            "assistant" => async_openai::types::ChatCompletionRequestMessage::Assistant(
                async_openai::types::ChatCompletionRequestAssistantMessage {
                    role: match self.role.as_ref() {
                        "user" => async_openai::types::Role::User,
                        "system" => async_openai::types::Role::System,
                        "assistant" => async_openai::types::Role::Assistant,
                        "function" => async_openai::types::Role::Function,
                        role => panic!("unsupported role '{role}'"),
                    },
                    content: self.content,
                    tool_calls: None,
                    function_call: self
                        .function_call
                        .map(|fc| async_openai::types::FunctionCall {
                            name: fc.name,
                            arguments: fc.arguments,
                        }),
                    name: self.name,
                },
            ),
            "function" => async_openai::types::ChatCompletionRequestMessage::Function(
                async_openai::types::ChatCompletionRequestFunctionMessage {
                    role: match self.role.as_ref() {
                        "user" => async_openai::types::Role::User,
                        "system" => async_openai::types::Role::System,
                        "assistant" => async_openai::types::Role::Assistant,
                        "function" => async_openai::types::Role::Function,
                        role => panic!("unsupported role '{role}'"),
                    },
                    content: self.content,
                    name: self.name.expect("function message name should be valid"),
                },
            ),
            role => panic!("unsupported role '{role}'"),
        }
    }
}

impl IntoChatCompletionRequestMessage for async_openai::types::ChatCompletionRequestMessage {
    #[inline]
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage {
        match self {
            async_openai::types::ChatCompletionRequestMessage::User(message) => {
                tiktoken_rs::ChatCompletionRequestMessage {
                    role: message.role.to_string(),
                    content: match message.content {
                        async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                            text,
                        ) => Some(text),
                        async_openai::types::ChatCompletionRequestUserMessageContent::Array(
                            array,
                        ) => {
                            if array.is_empty() {
                                None
                            } else {
                                panic!("unsupported user message content {array:?}")
                            }
                        }
                    },
                    function_call: None,
                    name: message.name,
                }
            }
            async_openai::types::ChatCompletionRequestMessage::System(message) => {
                tiktoken_rs::ChatCompletionRequestMessage {
                    role: message.role.to_string(),
                    content: Some(message.content),
                    function_call: None,
                    name: message.name,
                }
            }
            async_openai::types::ChatCompletionRequestMessage::Assistant(message) => {
                tiktoken_rs::ChatCompletionRequestMessage {
                    role: message.role.to_string(),
                    content: message.content,
                    function_call: message.function_call.map(|fc| tiktoken_rs::FunctionCall {
                        name: fc.name,
                        arguments: fc.arguments,
                    }),
                    name: message.name,
                }
            }
            async_openai::types::ChatCompletionRequestMessage::Function(message) => {
                tiktoken_rs::ChatCompletionRequestMessage {
                    role: message.role.to_string(),
                    content: message.content,
                    function_call: None,
                    name: Some(message.name),
                }
            }
            role @ async_openai::types::ChatCompletionRequestMessage::Tool(_) => {
                panic!("unsupported role '{role:?}'")
            }
        }
    }

    #[inline]
    fn into_async_openai(self) -> async_openai::types::ChatCompletionRequestMessage {
        self
    }
}

impl IntoChatCompletionRequestMessage for async_openai::types::ChatCompletionResponseMessage {
    #[inline]
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage {
        tiktoken_rs::ChatCompletionRequestMessage {
            role: self.role.to_string(),
            content: self.content,
            function_call: self.function_call.map(|fc| tiktoken_rs::FunctionCall {
                name: fc.name,
                arguments: fc.arguments,
            }),
            name: None,
        }
    }

    #[inline]
    fn into_async_openai(self) -> async_openai::types::ChatCompletionRequestMessage {
        match self.role {
            async_openai::types::Role::User => {
                async_openai::types::ChatCompletionRequestMessage::User(
                    async_openai::types::ChatCompletionRequestUserMessage {
                        role: self.role,
                        content: match self.content {
                            Some(text) => {
                                async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                                    text,
                                )
                            }
                            None => {
                                async_openai::types::ChatCompletionRequestUserMessageContent::Array(
                                    Vec::new(),
                                )
                            }
                        },
                        name: None,
                    },
                )
            }
            async_openai::types::Role::System => {
                async_openai::types::ChatCompletionRequestMessage::System(
                    async_openai::types::ChatCompletionRequestSystemMessage {
                        role: self.role,
                        content: self
                            .content
                            .expect("system message content should be valid"),
                        name: None,
                    },
                )
            }
            async_openai::types::Role::Assistant => {
                async_openai::types::ChatCompletionRequestMessage::Assistant(
                    async_openai::types::ChatCompletionRequestAssistantMessage {
                        role: self.role,
                        content: self.content,
                        tool_calls: None,
                        function_call: self.function_call,
                        name: None,
                    },
                )
            }
            async_openai::types::Role::Function => {
                async_openai::types::ChatCompletionRequestMessage::Function(
                    async_openai::types::ChatCompletionRequestFunctionMessage {
                        role: self.role,
                        content: self.content,
                        name: self
                            .function_call
                            .expect("function message function call should be valid")
                            .name,
                    },
                )
            }
            role @ async_openai::types::Role::Tool => panic!("unsupported role '{role}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let messages: Vec<async_openai::types::ChatCompletionRequestMessage> = Vec::new();

        assert_eq!(ChatSplitter::default().split(&messages).0, &[]);
        assert_eq!(ChatSplitter::default().split(&messages).1, &[]);
    }
}
