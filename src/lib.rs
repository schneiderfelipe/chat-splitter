use std::cmp::Ordering;

use indxvec::Search;
use tiktoken_rs::get_chat_completion_max_tokens;
use tiktoken_rs::model::get_context_size;

pub struct ChatSplitter {
    /// ID of the model to use.
    ///
    /// This is passed to `tiktoken-rs` to select the correct tokenizer for the
    /// model.
    model: String,

    /// The maximum number of tokens to leave for chat completion.
    ///
    /// This is the same as in the [official API](https://platform.openai.com/docs/api-reference/chat#completions/create-prompt) and given to `async-openai`.
    /// The total length of input tokens and generated tokens is limited by the
    /// model's context length. Splits will have at least that many tokens
    /// available for chat completion, never less.
    max_tokens: u16,

    /// The maximum number of messages to have in the chat.
    ///
    /// Splits will have at most that many messages, never more.
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
    #[inline]
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        let max_tokens = get_context_size(&model) / 2;
        let max_tokens = max_tokens as u16;

        let max_messages = MAX_MESSAGES_LIMIT / 2;

        Self {
            model,
            max_tokens,
            max_messages,
        }
    }

    #[inline]
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

    #[inline]
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

    #[inline]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Get a split position by only considering `max_messages`.
    #[inline]
    fn position_by_max_messages<M>(&self, messages: &[M], max_messages: usize) -> usize
    where
        M: IntoRequestMessage + Clone,
    {
        let upper_limit = max_messages.min(MAX_MESSAGES_LIMIT);

        let n = messages.len();
        let n = if n <= upper_limit { 0 } else { n - upper_limit };
        debug_assert!(messages[n..].len() <= upper_limit);
        n
    }

    /// Get a split position by only considering `max_tokens`.
    ///
    /// # Panics
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    fn position_by_max_tokens<M>(&self, messages: &[M], max_tokens: u16) -> usize
    where
        M: IntoRequestMessage + Clone,
    {
        let max_tokens = max_tokens as usize;
        let lower_limit = max_tokens.min(get_context_size(&self.model));

        let messages: Vec<_> = messages
            .iter()
            .cloned()
            .map(IntoRequestMessage::into_tiktoken_rs)
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

    /// Get the split position for the recent messages that fit the model's
    /// maximum number of tokens.
    ///
    /// This works by first considering the `max_messages` limit, then the
    /// `max_tokens` limit.
    ///
    /// # Panics
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    fn position<M>(&self, messages: &[M], max_tokens: u16, max_messages: usize) -> usize
    where
        M: IntoRequestMessage + Clone,
    {
        let n = self.position_by_max_messages(messages, max_messages);
        n + self.position_by_max_tokens(&messages[n..], max_tokens)
    }

    /// Get the most recent messages that fit the model's maximum number of
    /// tokens.
    ///
    /// # Panics
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    pub fn messages<'a, M>(&self, messages: &'a [M]) -> &'a [M]
    where
        M: IntoRequestMessage + Clone,
    {
        // TODO: transition to split
        let n = self.position(messages, self.max_tokens, self.max_messages);
        &messages[n..]
    }
}

pub trait IntoRequestMessage {
    /// Convert to `tiktoken-rs` completion request message.
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage;
    /// Convert to `async-openai` completion request message.
    fn into_async_openai(self) -> async_openai::types::ChatCompletionRequestMessage;
}

impl IntoRequestMessage for tiktoken_rs::ChatCompletionRequestMessage {
    #[inline]
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage {
        self
    }

    #[inline]
    fn into_async_openai(self) -> async_openai::types::ChatCompletionRequestMessage {
        async_openai::types::ChatCompletionRequestMessage {
            role: match self.role.as_ref() {
                "user" => async_openai::types::Role::User,
                "system" => async_openai::types::Role::System,
                "assistant" => async_openai::types::Role::Assistant,
                "function" => async_openai::types::Role::Function,
                role => panic!("unknown role '{role}'"),
            },
            content: self.content,
            function_call: self.function_call.map(|fc| {
                async_openai::types::FunctionCall {
                    name: fc.name,
                    arguments: fc.arguments,
                }
            }),

            name: self.name,
        }
    }
}

impl IntoRequestMessage for async_openai::types::ChatCompletionRequestMessage {
    #[inline]
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage {
        tiktoken_rs::ChatCompletionRequestMessage {
            role: self.role.to_string(),
            content: self.content,
            function_call: self.function_call.map(|fc| {
                tiktoken_rs::FunctionCall {
                    name: fc.name,
                    arguments: fc.arguments,
                }
            }),

            name: self.name,
        }
    }

    #[inline]
    fn into_async_openai(self) -> async_openai::types::ChatCompletionRequestMessage {
        self
    }
}

impl IntoRequestMessage for async_openai::types::ChatCompletionResponseMessage {
    #[inline]
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage {
        tiktoken_rs::ChatCompletionRequestMessage {
            role: self.role.to_string(),
            content: self.content,
            function_call: self.function_call.map(|fc| {
                tiktoken_rs::FunctionCall {
                    name: fc.name,
                    arguments: fc.arguments,
                }
            }),

            name: None,
        }
    }

    #[inline]
    fn into_async_openai(self) -> async_openai::types::ChatCompletionRequestMessage {
        async_openai::types::ChatCompletionRequestMessage {
            role: self.role,
            content: self.content,
            function_call: self.function_call,

            name: None,
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {}
}
