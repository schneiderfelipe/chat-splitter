use std::error::Error;

use async_openai::types::ChatCompletionRequestMessageArgs;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::types::Role;
use async_openai::Client;
use chat_memory::ChatSplitter;
use chat_memory::IntoRequestMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let max_messages = 10000u16;
    let max_tokens = 512u16;
    let model = "gpt-3.5-turbo";

    let memory_manager = ChatSplitter::default()
        .max_messages(max_messages)
        .max_tokens(max_tokens)
        .model(model);

    let mut messages = Vec::new();
    for _ in 0..1000 {
        messages.extend([
            ChatCompletionRequestMessageArgs::default()
                .role(Role::System)
                .content("You are a helpful assistant.")
                .build()?,
            ChatCompletionRequestMessageArgs::default()
                .role(Role::User)
                .content("Who won the world series in 2020?")
                .build()?,
            ChatCompletionRequestMessageArgs::default()
                .role(Role::Assistant)
                .content("The Los Angeles Dodgers won the World Series in 2020.")
                .build()?,
            ChatCompletionRequestMessageArgs::default()
                .role(Role::User)
                .content("Where was it played?")
                .build()?,
        ]);
    }

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(max_tokens)
        .model(model)
        .messages(memory_manager.messages(&messages))
        .build()?;

    let client = Client::new();
    let response = client.chat().create(request).await?;

    println!("\nResponse:\n");
    for choice in response.choices {
        println!(
            "{}: Role: {}  Content: {:?}",
            choice.index, choice.message.role, choice.message.content
        );
        messages.push(choice.message.into_async_openai());
    }

    println!("{:#?}", memory_manager.messages(&messages));
    Ok(())
}
