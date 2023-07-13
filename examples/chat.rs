use std::error::Error;

use async_openai::types::ChatCompletionRequestMessage;
use async_openai::types::ChatCompletionRequestMessageArgs;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::types::Role;
use async_openai::Client;
use chat_memory::ChatSplitter;
use chat_memory::IntoRequestMessage;

const MODEL: &str = "gpt-3.5-turbo";
const MAX_TOKENS: u16 = 1024;
const MAX_MESSAGES: usize = 16;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut previous_messages = previous_messages()?;
    previous_messages.push(
        ChatCompletionRequestMessageArgs::default()
            .role(Role::User)
            .content("Where was it played?")
            .build()?,
    );
    assert!(previous_messages.len() > MAX_MESSAGES);

    let memory_manager = ChatSplitter::new(MODEL)
        .max_tokens(MAX_TOKENS)
        .max_messages(MAX_MESSAGES);

    let mut messages = vec![ChatCompletionRequestMessageArgs::default()
        .role(Role::System)
        .content("You are a helpful assistant.")
        .build()?];
    messages.extend(memory_manager.messages(&previous_messages).iter().cloned());
    assert!(messages.len() <= MAX_MESSAGES + 1);

    let request = CreateChatCompletionRequestArgs::default()
        .model(MODEL)
        .max_tokens(MAX_TOKENS)
        .messages(messages)
        .build()?;

    let client = Client::new();
    let response = client.chat().create(request).await?;

    println!("\nResponse:\n");
    for choice in response.choices {
        println!(
            "{}: Role: {}  Content: {:?}",
            choice.index, choice.message.role, choice.message.content
        );

        previous_messages.push(choice.message.into_async_openai());
    }

    Ok(())
}

fn previous_messages() -> Result<Vec<ChatCompletionRequestMessage>, Box<dyn Error>> {
    let mut messages = Vec::new();
    for _ in 0..2000 {
        messages.extend([
            ChatCompletionRequestMessageArgs::default()
                .role(Role::User)
                .content("Who won the world series in 2020?")
                .build()?,
            ChatCompletionRequestMessageArgs::default()
                .role(Role::Assistant)
                .content("The Los Angeles Dodgers won the World Series in 2020.")
                .build()?,
        ]);
    }
    Ok(messages)
}
