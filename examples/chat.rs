use std::error::Error;

use async_openai::types::ChatCompletionRequestMessage;
use async_openai::types::ChatCompletionRequestMessageArgs;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::types::Role;
use async_openai::Client;
use chat_splitter::ChatSplitter;
use chat_splitter::IntoChatCompletionRequestMessage;

const MODEL: &str = "gpt-3.5-turbo";
const MAX_TOKENS: u16 = 1024;
const MAX_MESSAGES: usize = 16;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut stored_messages = get_stored_messages()?;
    stored_messages.push(
        ChatCompletionRequestMessageArgs::default()
            .role(Role::User)
            .content("Where was it played?")
            .build()?,
    );
    assert!(stored_messages.len() > MAX_MESSAGES);

    let (_outdated_messages, recent_messages) = ChatSplitter::new(MODEL)
        .max_tokens(MAX_TOKENS)
        .max_messages(MAX_MESSAGES)
        .split(&stored_messages);

    let mut messages = vec![ChatCompletionRequestMessageArgs::default()
        .role(Role::System)
        .content("You are a helpful assistant.")
        .build()?];
    messages.extend(recent_messages.iter().cloned());
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

        stored_messages.push(choice.message.into_async_openai());
    }

    Ok(())
}

fn get_stored_messages() -> Result<Vec<ChatCompletionRequestMessage>, Box<dyn Error>> {
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
