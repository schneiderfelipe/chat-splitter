use std::cell::Cell;
use std::error::Error;

use derive_builder::Builder;
use tiktoken_rs::get_chat_completion_max_tokens;

pub trait IntoRequestMessage {
    fn into_tiktoken_rs(self) -> tiktoken_rs::ChatCompletionRequestMessage;
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

const GOOD_INDEX_HINT: usize = 16;
const GOOD_MAX_MESSAGES: usize = 128;
const GOOD_MAX_TOKENS: usize = 1024;
const INDEX_LIMIT: usize = 2048;

#[derive(Builder, Debug, Default)]
#[builder(setter(into))]
pub struct ChatMemoryManager {
    /// ID of the model to use.
    ///
    /// This is used to select the correct tokenizer for the model.
    /// See [`CreateChatCompletionRequestArgs.model`].
    #[builder(default = "\"gpt-3.5-turbo\".into()")]
    model: String,

    /// The maximum number of tokens to leave for completion.
    ///
    /// This is to ensure there's room for the model to generate a response
    /// given the trimmed context.
    /// See [`CreateChatCompletionRequestArgs.max_tokens`].
    #[builder(default = "GOOD_MAX_TOKENS")]
    max_tokens: usize,

    /// The maximum number of messages to have in a conversation.
    #[builder(default = "GOOD_MAX_MESSAGES")]
    max_messages: usize,

    /// Index hint of the first message in the current context.
    ///
    /// This is updated as we go.
    #[builder(setter(skip), default = "GOOD_INDEX_HINT.into()")]
    index_hint: Cell<usize>,
}

impl ChatMemoryManager {
    /// Get the split position for the recent messages that fit the model's
    /// context window.
    ///
    /// # Errors
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    fn position<M>(&self, messages: &[M]) -> Result<usize, Box<dyn Error>>
    where
        M: IntoRequestMessage + Clone,
    {
        let messages: Vec<_> = messages
            .iter()
            .cloned()
            .map(IntoRequestMessage::into_tiktoken_rs)
            .collect();

        let last = |n| {
            let len = messages.len();
            let last = &messages[len - n..len];
            debug_assert_eq!(last.len(), n);
            last
        };
        let count = |n| -> Result<usize, Box<dyn Error>> {
            let count = get_chat_completion_max_tokens(&self.model, last(n))?;
            println!("{n} messages with {count} tokens");
            Ok(count)
        };
        let mut n = 0;
        while count(n)? > self.max_tokens {
            n += 1;
        }
        debug_assert!(get_chat_completion_max_tokens(&self.model, last(n))? > self.max_tokens);
        Ok(messages.len() - n)
    }

    /// Get the most recent messages that fit the model's context window.
    ///
    /// # Errors
    /// If tokenizer for the specified model is not found or is not a supported
    /// chat model.
    #[inline]
    pub fn messages<'a, M>(&self, messages: &'a [M]) -> Result<&'a [M], Box<dyn Error>>
    where
        M: IntoRequestMessage + Clone,
    {
        let n = self.position(messages)?;
        let len = messages.len();
        let last = &messages[len - n..len];
        debug_assert_eq!(last.len(), n);
        Ok(last)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {}
}
