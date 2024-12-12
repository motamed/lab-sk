import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
# Import the KernelArguments class
from semantic_kernel.functions import KernelArguments


async def main():
        
    kernel = Kernel()
    
    # Services
    chatcomp = AzureChatCompletion(
        deployment_name="gpt4o",
        base_url="https://demous.openai.azure.com/openai/",
        api_key="bf94c249d2b349378139731c33cbb522",
        api_version="2024-10-01-preview",
        service_id="gpt4o"
    )
    kernel.add_service(chatcomp)
    
    # Chat History
    chatHistory = ChatHistory()
    
    ##################################### Plugins #####################################
    
    # using plugins from the prompts folder
    promptDir = "./prompts"
    kernel.add_plugin(parent_directory=promptDir, plugin_name="SummarizePlugin")
    
    
    # Add the TranslatePlugin 
    
    
    ##################################### Plugins #####################################
    
    
    # Planner
    execution_settings = AzureChatPromptExecutionSettings(service_id="gpt4o",tool_choice="auto")
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    # Kernel Arguments
    arguments = KernelArguments(settings=execution_settings)
    
    # Chat Loop
    userInput = None
    while True:
        userInput = input("You: ")

        if userInput == "exit":
            break
        
        # Add the user input to the the template arguments
        arguments["input"] = userInput
            
        chatHistory.add_user_message(userInput)
        result = await chatcomp.get_chat_message_content(
            chat_history=chatHistory, kernel=kernel, settings=execution_settings
        )
        print("Assistant > " + str(result))
        chatHistory.add_message(result)  
        
# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
