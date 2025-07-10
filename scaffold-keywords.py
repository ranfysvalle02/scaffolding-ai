import datetime
import os
import re

# --- Core Code Components ---

# A dictionary to hold all the building blocks for our agent.
CODE_COMPONENTS = {
    "imports": """# Import required modules
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from langgraph.store.mongodb import MongoDBStore
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
""",
    "env_loader": '''# Load environment variables from .env file
load_dotenv()
''',
    "embedding_model": '''# Initialize embedding model
embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2025-01-01-preview",
)
''',
    "llm": '''# Initialize language model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
    model="gpt-4o",
)
''',
    "mongo_client": '''# Initialize MongoDB client
mongo_conn_str = "mongodb://localhost:27017?retryWrites=true&w=majority&directConnection=true"
mongo_client = MongoClient(mongo_conn_str)
''',
    "checkpointer": '''# Initialize checkpointer - used to save short-term conversation state
checkpointer = MongoDBSaver(
    client=mongo_client, db_name="langgraph_db", collection_name="checkpoints"
)
''',
    "store": '''# Initialize Store - used for long-term memory to save/retrieve memories
store = MongoDBStore(
    collection=Collection(
        database=Database(name="long-term-memory", client=mongo_client),
        name="my-memories",
    ),
    index_config={
        "embed": embedding_model,  # Your embedding model
        "dims": len(embedding_model.embed_query("")),  # Embedding dimensions
        "fields": ["$"],  # Fields to index
        "filters": None,  # The required filters key
    },
)
''',
    "tools": '''# Initialize tools for the agent
tools = [
    create_manage_memory_tool(namespace="memories"),
    create_search_memory_tool(namespace="memories"),
]
''',
    "agent": '''# Create the agent
agent = create_react_agent(
    llm,
    tools=tools,
    store=store,
    checkpointer=checkpointer,
)
''',
    "conversation": '''# Start conversation
config = {"configurable": {"thread_id": "my-thread-1"}}
# Using a new thread_id for a clean start
response = agent.invoke({"messages": "My name is John and MongoDB is my favorite database."}, config=config)

# Print the conversation messages
for message in response["messages"]:
    message.pretty_print()
''',
    "note": '''# Note:
# If you are using Python 3.10 or older, please upgrade to Python 3.11 or newer.
# The langmem library requires a feature available only in Python 3.11 and newer.
'''
}

# --- LLM-Powered Generator ---

def generate_code_from_prompt(prompt: str):
    """
    Generates a Python script based on a natural language prompt by selecting
    and combining relevant code components.

    Args:
        prompt: A string describing the desired code components.
    """
    prompt = prompt.lower()
    selected_components = []
    
    # Define keywords to look for in the prompt for each component
    keyword_map = {
        "llm": ["llm", "language model"],
        "embedding": ["embedding", "embedder"],
        "mongo": ["mongo", "database", "mongodb"],
        "checkpointer": ["checkpointer", "mongodbsaver", "saver"],
        "store": ["store", "mongodbstore", "memory"],
        "tools": ["tools", "memory tool"],
        "agent": ["agent", "react agent"],
        "conversation": ["conversation", "invoke", "run"],
        "full": ["full", "all", "everything", "bells and whistles"]
    }

    # A simple LLM-like logic to determine which components to include
    if any(keyword in prompt for keyword in keyword_map["full"]):
        # If a "full" keyword is found, include everything needed for a complete agent
        selected_components = [
            "imports", "env_loader", "embedding_model", "llm", "mongo_client",
            "checkpointer", "store", "tools", "agent", "conversation", "note"
        ]
    else:
        # Add components based on individual keywords
        if any(keyword in prompt for keyword in keyword_map["llm"]):
            selected_components.extend(["imports", "env_loader", "llm"])
        if any(keyword in prompt for keyword in keyword_map["store"]):
             selected_components.extend(["imports", "env_loader", "embedding_model", "mongo_client", "store"])
        if any(keyword in prompt for keyword in keyword_map["checkpointer"]):
            selected_components.extend(["imports", "mongo_client", "checkpointer"])
        if any(keyword in prompt for keyword in keyword_map["agent"]):
            # Agent requires many other components, so we add its dependencies
            selected_components.extend([
                "imports", "env_loader", "embedding_model", "llm", "mongo_client",
                "checkpointer", "store", "tools", "agent"
            ])
        # Add other components if requested...
    
    # Ensure 'imports' is always first if any other component is selected
    if selected_components and "imports" not in selected_components:
        selected_components.insert(0, "imports")
    
    # Remove duplicates while preserving order
    final_components = list(dict.fromkeys(selected_components))
    
    if not final_components:
        print("I'm sorry, I couldn't determine which components you need. Please be more specific.")
        return

    # --- File Generation ---
    # Create a filename based on the selected components
    base_name = "_".join(re.sub(r'(model|client|loader)', '', name) for name in final_components if name != 'imports')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.py" if base_name else f"scaffold_{timestamp}.py"

    # Assemble the final code
    final_code = "\n\n".join(CODE_COMPONENTS[key] for key in final_components)
    
    with open(filename, 'w') as file:
        file.write(final_code)
        
    print(f"✨ Scaffold code has been successfully written to {filename}")

# --- Example Usage ---

if __name__ == "__main__":
    generate_code_from_prompt("Give me a full react agent with all the bells and whistles")
