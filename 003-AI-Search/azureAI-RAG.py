# Copyright (c) Microsoft. All rights reserved.
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
# from semantic_kernel.core_plugins import TextMemoryPlugin
# from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore

##################################### Azure AI Search #####################################
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchCollection, AzureAISearchStore
from semantic_kernel.data.vector_search import VectorSearchOptions
from datamodel import HotelSampleClass



async def main():
    
    
    ##################################### Azure AI Search #####################################
    store = AzureAISearchStore(api_key="WtZM5SSHQg4CNtDBsl6doKfTtI5zBEB9Lem7PKweriAzSeB0muPW",search_endpoint="https://aoai.search.windows.net")
    collection: AzureAISearchCollection = store.get_collection("hotels-sample-index", HotelSampleClass)

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
    
    # Text search
    query = "swimming pool and good internet connection"
    results = await collection.text_search(search_text=query)
    print("Search results using text: ")
    async for result in results.results:
        print(
            f"    {result.record.HotelId} (in {result.record.Address['City']}, "
            f"{result.record.Address['Country']}): {result.record.Description} (score: {result.score})"
        )

    # memory = SemanticTextMemory(
    #     storage=VolatileMemoryStore(), embeddings_generator=embedding_gen
    # )
    # kernel.add_plugin(TextMemoryPlugin(memory), "memory")

    # await memory.save_information(
    #     collection="generic", id="info1", text="My budget for 2024 is $100,000"
    # )

    # result = await kernel.invoke_prompt(
    #     function_name="budget",
    #     plugin_name="BudgetPlugin",
    #     prompt="{{memory.recall 'budget by year'}} What is my budget for 2024?",
    # )
    # print(result)

    # # Generate the vector for the query
    # query_vector = (await embedding_gen.generate_raw_embeddings([query]))[0]

    # print("Search results using vector: ")
    # # Use vectorized search to search using the vector.
    # results = await collection.vectorized_search(
    #     vector=query_vector
    # )
    # async for result in results.results:
    #     print(
    #         f"    {result.record.HotelId} (in {result.record.Address['City']}, "
    #         f"{result.record.Address['Country']}): {result.record.Description} (score: {result.score})"
    #     )

    # # Delete the collection object so that the connection is closed.
    del collection
    await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
