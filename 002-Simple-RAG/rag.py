# Copyright (c) Microsoft. All rights reserved.
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore

async def main():

    kernel = Kernel()

    chatcomp = AzureChatCompletion(
        deployment_name="gpt4o",
        base_url="https://demous.openai.azure.com/openai/",
        api_key="bf94c249d2b349378139731c33cbb522",
        api_version="2024-10-01-preview",
        service_id="gpt4o",
    )

    kernel.add_service(chatcomp)
    embedding_gen = AzureTextEmbedding(
        api_key="bf94c249d2b349378139731c33cbb522",
        base_url="https://demous.openai.azure.com/openai/",
        deployment_name="text-embedding-ada-002",
    )

    kernel.add_service(embedding_gen)

    memory = SemanticTextMemory(
        storage=VolatileMemoryStore(), embeddings_generator=embedding_gen
    )
    
    ### Add the memory plugin to the kernel
    

    await memory.save_information(
        collection="generic", id="info1", text="My budget for 2024 is $100,000"
    )

    result = await kernel.invoke_prompt(
        function_name="budget",
        plugin_name="BudgetPlugin",
        prompt="{{memory.recall 'budget by year'}} What is my budget for 2024?",
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
