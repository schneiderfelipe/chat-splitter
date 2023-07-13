use indxvec::Search;
use tiktoken_rs::get_chat_completion_max_tokens;

pub struct ChatSplitter {
    /// ID of the model to use.
    ///
    /// This is used to select the correct tokenizer for the model.
    /// See [`CreateChatCompletionRequestArgs.model`].
    model: String,

    /// The maximum number of tokens to leave for completion.
    ///
    /// This is to ensure there's room for the model to generate a response
    /// given the trimmed context.
    /// See [`CreateChatCompletionRequestArgs.max_tokens`].
    max_tokens: usize,

    /// The maximum number of messages to have in a conversation context.
    max_messages: usize,
}

const DEFAULT_MAX_MESSAGES: usize = 128;
const DEFAULT_MAX_TOKENS: usize = 1_024;
const DEFAULT_MODEL: &str = "gpt-3.5-turbo";
const MAX_MESSAGES_LIMIT: usize = 2_048;
const MAX_TOKENS_LIMIT: usize = 32_768;

impl Default for ChatSplitter {
    #[inline]
    fn default() -> Self {
        Self {
            max_messages: DEFAULT_MAX_MESSAGES,
            max_tokens: DEFAULT_MAX_TOKENS,
            model: DEFAULT_MODEL.into(),
        }
    }
}

impl ChatSplitter {
    #[inline]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    #[inline]
    pub fn max_messages(mut self, max_messages: impl Into<usize>) -> Self {
        self.max_messages = max_messages.into();
        self
    }

    #[inline]
    pub fn max_tokens(mut self, max_tokens: impl Into<usize>) -> Self {
        self.max_tokens = max_tokens.into();
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
        let limit = max_messages.min(MAX_MESSAGES_LIMIT);
        let n = messages.len();

        if n <= limit {
            0
        } else {
            n - limit
        }
    }

    /// Get a split position by only considering `max_tokens`.
    ///
    /// # Panics
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    fn position_by_max_tokens<M>(&self, messages: &[M], max_tokens: usize) -> usize
    where
        M: IntoRequestMessage + Clone,
    {
        let messages: Vec<_> = messages
            .iter()
            .cloned()
            .map(IntoRequestMessage::into_tiktoken_rs)
            .collect();

        let limit = max_tokens.min(MAX_TOKENS_LIMIT);
        let (n, range) = (0..=messages.len() - 1).binary_any(|m| {
            let tokens = get_chat_completion_max_tokens(&self.model, &messages[m..])
                .expect("tokenizer should be available");
            tokens.cmp(&limit).reverse()
        });
        dbg!(n);
        dbg!(range);
        debug_assert!(
            get_chat_completion_max_tokens(&self.model, &messages[n..])
                .expect("tokenizer should be available")
                >= max_tokens
        );
        n
    }

    /// Get the split position for the recent messages that fit the model's
    /// context window.
    ///
    /// This works by first considering the `max_messages` limit, then the
    /// `max_tokens` limit.
    ///
    /// # Panics
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    fn position<M>(&self, messages: &[M], max_messages: usize, max_tokens: usize) -> usize
    where
        M: IntoRequestMessage + Clone,
    {
        let n = self.position_by_max_messages(messages, max_messages);
        n + self.position_by_max_tokens(&messages[n..], max_tokens)
    }

    /// Get the most recent messages that fit the model's context window.
    ///
    /// # Panics
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    pub fn messages<'a, M>(&self, messages: &'a [M]) -> &'a [M]
    where
        M: IntoRequestMessage + Clone,
    {
        let n = self.position(messages, self.max_messages, self.max_tokens);
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
