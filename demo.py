import datetime
import os
import re
import json
import getpass

# --- Core Code Components (Derived from LangChain Documentation) ---
# This dictionary holds all the building blocks for our agent.
# The LLM (me) selects from this "context" to build the script.
CODE_COMPONENTS = {
    "imports": """# Import required modules from langchain and other libraries
import os
import getpass
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.store.mongodb import MongoDBStore
from langgraph.checkpoint.mongodb import MongoDBSaver
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
""",
    "env_loader": '''# Load environment variables from .env file or prompt if not set
load_dotenv()
# Set credentials based on documentation best practices
if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter your Azure OpenAI API key: ")
if "AZURE_OPENAI_ENDPOINT" not in os.environ:
    os.environ["AZURE_OPENAI_ENDPOINT"] = input("Enter your Azure OpenAI Endpoint (e.g., https://your-endpoint.openai.azure.com/): ")
''',
    "llm": '''# Initialize language model as per the documentation
# Replace 'azure_deployment' with the name of your chat model deployment
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",
    api_version="2023-06-01-preview",
    temperature=0,
    max_retries=2,
)
''',
    "embedding_model": '''# Initialize embedding model
# Replace 'azure_deployment' with the name of your embedding model deployment
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    api_version="2023-06-01-preview",
)
''',
    "mongo_client": '''# Initialize MongoDB client
mongo_conn_str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(mongo_conn_str)
''',
    "checkpointer": '''# Initialize checkpointer for saving conversation state
checkpointer = MongoDBSaver(
    client=mongo_client, db_name="langgraph_db", collection_name="checkpoints"
)
''',
    "store": '''# Initialize Store for long-term memory
store = MongoDBStore(
    collection=Collection(
        database=Database(name="long-term-memory", client=mongo_client),
        name="my-memories",
    ),
    index_config={
        "embed": embedding_model,
        "dims": len(embedding_model.embed_query("test")),
        "fields": ["$"],
        "filters": None,
    },
)
''',
    "tools": '''# Initialize tools for the agent to use memory
tools = [
    create_manage_memory_tool(namespace="memories"),
    create_search_memory_tool(namespace="memories"),
]
''',
    "agent": '''# Create the REACT agent
agent = create_react_agent(
    llm,
    tools=tools,
    store=store,
    checkpointer=checkpointer,
)
''',
    "agent_invocation": '''# --- Agent Invocation Example ---
print("\\n--- Running Agent Invocation ---")
config = {"configurable": {"thread_id": "thread-1"}}
# The agent uses its tools and memory here
print("User: My name is Bob. I live in Paris.")
agent.invoke({"messages": "My name is Bob. I live in Paris."}, config=config)
print("User: What is my name?")
response = agent.invoke({"messages": "What is my name?"}, config=config)
print("\\nFinal response from agent:")
response["messages"][-1].pretty_print()
''',
    "note": '''# Note:
# The langmem library requires Python 3.11 or newer.
# Ensure your .env file has AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.
'''
}

# --- LLM-Powered Generator ---

def get_llm_scaffolding_plan(prompt: str) -> dict:
    """
    This function simulates an LLM generating a structured plan (JSON)
    to fulfill a scaffolding request based on available components.
    """
    prompt = prompt.lower()
    plan = {
        "request_analysis": {"prompt": prompt},
        "components_selected": []
    }
    # This is where the "LLM" makes its decisions.
    if any(p in prompt for p in ["full", "all", "everything", "cmon man", "almost"]):
        plan["request_analysis"]["detected_intent"] = "Generate a complete, runnable example for a LangGraph REACT agent with memory."
        plan["request_analysis"]["dependencies"] = "A full agent requires: imports, environment loading, LLM/embedding models, a vector store, a checkpointer, tools, the agent definition, and a runnable example."
        plan["components_selected"] = [
            "imports", "env_loader", "llm", "embedding_model", "mongo_client",
            "checkpointer", "store", "tools", "agent", "agent_invocation", "note"
        ]
    # This path is now for defining an agent without the runnable example
    elif "agent" in prompt:
        plan["request_analysis"]["detected_intent"] = "Generate the code for a REACT agent without a runnable example."
        plan["components_selected"] = [
            "imports", "env_loader", "llm", "embedding_model", "mongo_client",
            "checkpointer", "store", "tools", "agent"
        ]
    else:
        plan["request_analysis"]["detected_intent"] = "Could not determine a clear intent. Defaulting to a basic LLM instantiation."
        plan["components_selected"] = ["imports", "env_loader", "llm"]
    return plan

def generate_code_from_prompt(prompt: str):
    """
    Generates a Python script based on a natural language prompt by using an
    LLM-generated JSON plan to select and combine relevant code components.
    """
    print("🤖 Generating scaffolding plan from prompt...")
    llm_plan = get_llm_scaffolding_plan(prompt)
    print("✅ Plan generated. Reasoning:")
    print(json.dumps(llm_plan, indent=2))
    final_components = llm_plan.get("components_selected", [])
    if not final_components:
        print("\n🤔 LLM could not determine which components are needed. Please be more specific.")
        return

    # --- File Generation ---
    intent_slug = re.sub(r'[^a-z0-9_]', '', llm_plan["request_analysis"]["detected_intent"].lower().replace(" ", "_"))[:50]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scaffold_{intent_slug}_{timestamp}.py"
    
    # Assemble the final code with the corrected newline joiner
    final_code = "\n\n".join(CODE_COMPONENTS[key] for key in final_components)
    
    with open(filename, 'w') as file:
        file.write(final_code)
    print(f"\n✨ Scaffold code has been successfully written to {filename}")

# --- Example Usage ---

if __name__ == "__main__":
    generate_code_from_prompt("Generate the full script for a complete agent with memory.")
