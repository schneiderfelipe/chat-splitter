use async_openai::types::ChatCompletionRequestMessage;
use async_openai::types::ChatCompletionResponseMessage;

#[derive(Clone, Debug, Default)]
pub struct ChatMemory {
    /// ID of the model to use.
    ///
    /// This is used to select the correct tokenizer for the model.
    /// See [`CreateChatCompletionRequestArgs.model`].
    model: String,

    /// The maximum number of tokens to leave for completion.
    ///
    /// This is to ensure there's room for the model to generate a response
    /// given the trimmed context. See [`CreateChatCompletionRequestArgs.
    /// max_tokens`].
    max_tokens: u16,

    /// Chat messages in memory.
    messages: Vec<ChatCompletionRequestMessage>,
}

const N: usize = 2048;

impl ChatMemory {
    #[inline]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    #[inline]
    pub fn max_tokens(mut self, max_tokens: impl Into<u16>) -> Self {
        self.max_tokens = max_tokens.into();
        self
    }

    #[inline]
    pub fn push(&mut self, message: impl Message) {
        self.messages.push(message.message());
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    #[inline]
    #[must_use]
    pub fn messages(&self) -> &[ChatCompletionRequestMessage] {
        let messages = match self.len() {
            n if n < N => self.as_ref(),
            n => &self.as_ref()[n - N..n],
        };

        // TODO: correct for token count
        messages
    }
}

impl AsRef<[ChatCompletionRequestMessage]> for ChatMemory {
    #[inline]
    fn as_ref(&self) -> &[ChatCompletionRequestMessage] {
        &self.messages
    }
}

impl AsMut<[ChatCompletionRequestMessage]> for ChatMemory {
    #[inline]
    fn as_mut(&mut self) -> &mut [ChatCompletionRequestMessage] {
        &mut self.messages
    }
}

impl<M> Extend<M> for ChatMemory
where
    M: Message,
{
    #[inline]
    fn extend<T: IntoIterator<Item = M>>(&mut self, messages: T) {
        self.messages
            .extend(messages.into_iter().map(Message::message));
    }
}

pub trait Message {
    fn message(self) -> ChatCompletionRequestMessage;
}

impl Message for ChatCompletionRequestMessage {
    #[inline]
    fn message(self) -> ChatCompletionRequestMessage {
        self
    }
}

impl Message for ChatCompletionResponseMessage {
    #[inline]
    fn message(self) -> ChatCompletionRequestMessage {
        ChatCompletionRequestMessage {
            role: self.role,
            content: self.content,
            name: None,
            function_call: self.function_call,
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {}
}
