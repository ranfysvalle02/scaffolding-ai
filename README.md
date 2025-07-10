# scaffolding-ai

----

# Build a Better Workflow: Scaffolding LangGraph Agents with Python

We've all been there. You have a great idea for an AI agent, but before you can write a single line of innovative code, you're stuck doing chores: creating the project structure, managing imports, loading API keys, and writing the same database connection logic for the tenth time. This initial setup, or boilerplate, is a universal drag on developer momentum.

What if you could automate that entire process? This post explores a practical Python script that does just that. It's a scaffolding tool that builds a complete, ready-to-run LangGraph agent from a simple text prompt. It’s a powerful way to streamline your workflow, enforce best practices, and get straight to the creative work of building intelligent applications.

-----

### The "Why": From Repetitive Task to Reusable Tool

The script was born from a common developer pain point: the tedious repetition involved in starting new projects. The goal was to create a tool that could:

  * **Automate** the creation of all standard setup code.
  * **Enforce** a consistent, best-practices architecture.
  * **Accelerate** the time from idea to working prototype.

By embedding expert knowledge and best practices directly into the generator, you ensure every new project starts on a solid foundation.

-----

### How It Works: An Automated Assembly Line for Code

The script operates like a smart assembly line. It uses a library of pre-written code "components" and a simple "planner" to piece them together based on your instructions.

  * **`CODE_COMPONENTS`:** This Python dictionary is the library of parts. It contains string-based snippets for every conceivable piece of the agent: imports, environment loading, LLM initialization, database connections, memory stores, and even a working example.
  * **`get_llm_scaffolding_plan()`:** This function acts as the planner. It analyzes your prompt (e.g., "Generate the full script") and creates a JSON plan that lists exactly which components are needed.
  * **`generate_code_from_prompt()`:** This is the builder. It takes the plan, fetches the code for each component, joins them in the correct order, and writes the final, polished script to a new `.py` file.

Here is the full code for the generator itself, which showcases how these pieces fit together.

#### The Generator Script

```python
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
```

-----

### The Result: A Tour of the Generated Agent

Running the generator produces a clean, runnable Python script. This isn't just a collection of functions; it's a complete, stateful application with memory.

#### The Generated Agent Script

```python
# Import required modules from langchain and other libraries
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


# Load environment variables from .env file or prompt if not set
load_dotenv()
# Set credentials based on documentation best practices
if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter your Azure OpenAI API key: ")
if "AZURE_OPENAI_ENDPOINT" not in os.environ:
    os.environ["AZURE_OPENAI_ENDPOINT"] = input("Enter your Azure OpenAI Endpoint (e.g., https://your-endpoint.openai.azure.com/): ")


# Initialize language model as per the documentation
# Replace 'azure_deployment' with the name of your chat model deployment
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",
    api_version="2023-06-01-preview",
    temperature=0,
    max_retries=2,
)


# Initialize embedding model
# Replace 'azure_deployment' with the name of your embedding model deployment
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    api_version="2023-06-01-preview",
)


# Initialize MongoDB client
mongo_conn_str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(mongo_conn_str)


# Initialize checkpointer for saving conversation state
checkpointer = MongoDBSaver(
    client=mongo_client, db_name="langgraph_db", collection_name="checkpoints"
)


# Initialize Store for long-term memory
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


# Initialize tools for the agent to use memory
tools = [
    create_manage_memory_tool(namespace="memories"),
    create_search_memory_tool(namespace="memories"),
]


# Create the REACT agent
agent = create_react_agent(
    llm,
    tools=tools,
    store=store,
    checkpointer=checkpointer,
)


# --- Agent Invocation Example ---
print("\n--- Running Agent Invocation ---")
config = {"configurable": {"thread_id": "thread-1"}}
# The agent uses its tools and memory here
print("User: My name is Bob. I live in Paris.")
agent.invoke({"messages": "My name is Bob. I live in Paris."}, config=config)
print("User: What is my name?")
response = agent.invoke({"messages": "What is my name?"}, config=config)
print("\nFinal response from agent:")
response["messages"][-1].pretty_print()


# Note:
# The langmem library requires Python 3.11 or newer.
# Ensure your .env file has AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.

```

**Key Features of the Generated Script:**

  * **Robust Setup:** It handles all imports and securely loads credentials.
  * **Stateful Memory:** It configures MongoDB for both short-term conversation state (`MongoDBSaver`) and long-term semantic memory (`MongoDBStore`). This allows the agent to remember facts across sessions.
  * **Tool Use:** The agent is equipped with tools to explicitly manage and search its memory, a key feature of advanced agent architectures.
  * **Runnable Example:** It includes a simple, clear test case that immediately demonstrates the agent's memory capabilities, confirming the entire stack is working correctly.

-----

### The Payoff: More Than Just Saving Time

Adopting a scaffolding approach provides several powerful advantages that go beyond simple convenience.

  * **Accelerated Prototyping:** The most significant benefit is speed. You can test new ideas instantly, moving from concept to a working, stateful agent in seconds.
  * **Architectural Consistency:** For teams, this is crucial. The generator ensures every developer builds on the same tested, reliable, and standardized foundation, which dramatically reduces integration issues.
  * **Reduced Cognitive Overhead:** By automating the mundane, you free up mental energy to focus on the complex, creative problems that provide real business value—like the agent's core logic and unique skills.
  * **Simplified Onboarding:** New team members can get up to speed in record time. By generating a working, best-practices agent, they have an immediate, interactive learning tool to explore the project's architecture.

By building a tool to solve your own repetitive problems, you create a lasting asset that pays dividends in speed, quality, and innovation on every subsequent project.

-----

# APPENDIX


### 1\. Prompt to Generate Only the Agent Definition

This prompt focuses on creating the agent itself but excludes the runnable example at the end. This is useful if you want to import this agent definition into another part of a larger application.

**Sample Prompt:**

```python
generate_code_from_prompt("Just give me the agent definition code.")
```

**What it does:**
The keyword `"agent"` triggers the second `elif` block in the plan. The generated script will include:

  * `imports`
  * `env_loader`
  * `llm` and `embedding_model` initialization
  * `mongo_client` connection
  * `checkpointer` and `store` for memory
  * `tools`
  * The `agent` creation itself

It will **not** include the `agent_invocation` block, so the script defines the agent but doesn't run it.

-----

### 2\. Prompt for a Basic LLM Setup

This is the most fundamental request. It's perfect if you just need to start a script that connects to the language model and nothing else.

**Sample Prompt:**

```python
generate_code_from_prompt("I just need to instantiate the LLM.")
```

*(Other similar prompts that would work: "llm please", "basic setup")*

**What it does:**
Since this prompt doesn't contain "full" or "agent", it falls through to the `else` condition, which is the default, most basic plan. The generated script will only contain:

  * `imports` (a subset would technically be needed, but the plan includes all for simplicity)
  * `env_loader` for API keys
  * `llm` initialization

This gives you a barebones script to start interacting with the Azure OpenAI model immediately.

-----

### How these prompts work with the script's logic:

The `get_llm_scaffolding_plan` function in your script uses simple keyword matching to decide which components to include:

1.  If the prompt contains words like `"full"`, `"all"`, or `"everything"`, it builds the **complete, runnable script**.
2.  If not, it checks if the prompt contains the word `"agent"`. If so, it builds the **agent definition without the runnable example**.
3.  If neither of the above conditions is met, it defaults to the **most basic LLM setup**.

These simpler prompts allow you to generate only the specific parts of the architecture you need, making the scaffolding tool more flexible.
