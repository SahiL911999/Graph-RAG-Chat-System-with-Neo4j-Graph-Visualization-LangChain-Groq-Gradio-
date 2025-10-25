
"""
### Environment Setup and Imports

This section initializes the environment and imports the necessary modules and classes for the notebook. It includes loading environment variables, importing libraries for working with Neo4j, LangChain, and other utilities for processing and querying data.

The following key imports and initializations are performed:
- `load_dotenv`: Loads environment variables from a `.env` file.
- `Neo4jGraph`: For interacting with Neo4j databases.
- LangChain components such as `RunnableBranch`, `RunnableLambda`, `ChatPromptTemplate`, and `LLMGraphTransformer`.
- `ChatGroq` and `ChatOpenAI`: For using Groq and OpenAI models.
- Other utilities like `WikipediaLoader`, `TokenTextSplitter`, and `PyPDFLoader` for document processing.

Additionally, the `ChatGroq` instance is initialized with specific parameters for temperature and model name.
"""

from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_groq import ChatGroq

from langchain_community.document_loaders import CSVLoader

load_dotenv()

chat = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

"""### Environment Variables Setup for Neo4j

This section defines and initializes the necessary environment variables for the notebook. These variables are used to configure connections to external services such as Neo4j and OpenAI.

The following environment variables are set:
- `AURA_INSTANCENAME`: The name of the Neo4j Aura instance.
- `NEO4J_URI`: The URI for connecting to the Neo4j database.
- `NEO4J_USERNAME`: The username for authenticating with the Neo4j database.
- `NEO4J_PASSWORD`: The password for authenticating with the Neo4j database.
- `AUTH`: A tuple containing the Neo4j username and password.
- `OPENAI_API_KEY`: The API key for accessing OpenAI services.

"""

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
) #database=NEO4J_DATABASE,

"""### Data Processing and Document Loading

This section processes the PDF files or CSV files (need to change the loader). 

1. Loads the file
2. Splits the document into manageable chunks using RecursiveCharacterTextSplitter with:
    - Chunk size: 1200 characters
    - Overlap: 200 characters
3. Prints the total number of document chunks created

The processed documents will be used for further analysis and querying.

"""

raw_documents = CSVLoader("IY2.csv").load()
print(len(raw_documents))

"""## Below cell performs document processing and graph transformation using LangChain's LLMGraphTransformer. Here's what the code does:

1. Imports necessary libraries:
    - ThreadPoolExecutor for parallel processing
    - tqdm for progress tracking
    - pickle for serialization

2. Initializes LLMGraphTransformer with the chat model

3. Defines a helper function `process_document` that converts a single document to graph format

4. Processes documents in parallel batches of 100 using ThreadPoolExecutor:
    - Submits each document in the batch for processing
    - Collects results and extends the graph_documents list
    - Shows progress with tqdm

5. Saves the processed graph documents to a pickle file for later use

The processing leverages multithreading to speed up the graph transformation of the documents while providing visual progress feedback.

"""

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle

llm_transformer = LLMGraphTransformer(llm=chat)

# Function to process a single document
def process_document(doc):
    return llm_transformer.convert_to_graph_documents([doc])

# Path to pickle
PICKLE_PATH = "graph_documents.pkl"

if os.path.exists(PICKLE_PATH):
    # Load existing pickle
    with open(PICKLE_PATH, "rb") as f:
        graph_documents = pickle.load(f)
    print(f"Loaded {len(graph_documents)} graph documents from pickle.")
else:
    # If missing, process and save
    llm_transformer = LLMGraphTransformer(llm=chat)

    def process_document(doc):
        return llm_transformer.convert_to_graph_documents([doc])

    graph_documents = []
    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(documents), desc="Processing Documents") as pbar:
            for i in range(0, len(documents), 100):
                batch = documents[i : i + 100]
                futures = {executor.submit(process_document, doc): doc for doc in batch}
                for future in futures:
                    result = future.result()
                    graph_documents.extend(result)
                    pbar.update(1)

    # Save to pickle for next runs
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(graph_documents, f)
    print(f"Processed and saved {len(graph_documents)} graph documents to pickle.")

"""### Load Graph Documents

This cell loads the previously processed graph documents from a pickle file and displays the total number of loaded documents. These documents contain the structured graph representation of the annual report data.

"""

"""### Neo4j Graph Storage

This cell stores the processed graph documents to the Neo4j database. The following operations are performed:

1. Uses the Neo4jGraph instance (`kg`) to add the graph documents
2. Includes source information for traceability
3. Uses base entity labels for node classification

The documents contain structured information extracted from the annual report that will be used for graph-based querying and analysis.
"""

import pickle
import time
import os

def setup_graph_database():
    """Complete setup with all optimizations"""
    
    pickle_path = "graph_documents.pkl"
    
    # Step 1: Check if pickle exists, if not create it
    if not os.path.exists(pickle_path):
        print("❌ Pickle file not found. Please run the document processing first.")
        return False
    
    # Step 2: Load pickle file
    print("📂 Loading graph documents from pickle...")
    with open(pickle_path, "rb") as f:
        graph_documents = pickle.load(f)
    print(f"✅ Loaded {len(graph_documents)} graph documents.")
    
    # Step 3: Check if data already exists in Neo4j
    try:
        result = kg.query("MATCH (n:Document) RETURN count(n) as doc_count")
        existing_docs = result[0]["doc_count"] if result else 0
        
        if existing_docs > 0:
            print(f"✅ Found {existing_docs} existing Document nodes. Skipping graph loading.")
            return True
        
    except Exception as e:
        print(f"⚠️  Error checking existing data: {e}")
    
    # Step 4: Load data into Neo4j
    print("📥 Loading graph data into Neo4j (this may take a moment)...")
    start_time = time.time()
    
    try:
        res = kg.add_graph_documents(
            graph_documents,
            include_source=True,
            baseEntityLabel=True,
        )
        
        end_time = time.time()
        print(f"✅ Successfully loaded graph data in {end_time - start_time:.2f} seconds.")
        return True
        
    except Exception as e:
        print(f"❌ Error loading graph data: {e}")
        return False

# Use the optimized setup
if setup_graph_database():
    print("🚀 Graph database is ready!")
else:
    print("❌ Graph database setup failed!")

print(kg.schema)

"""### Entity Extraction from Text

This cell defines a class `Entities` to extract organization and person entities from a given text. It uses a `ChatPromptTemplate` to structure the prompt for entity extraction and invokes the `entity_chain` to process the input question. The extracted entities are printed as the output.

The following steps are performed:
1. Define the `Entities` class to structure the extracted entity information.
2. Create a `ChatPromptTemplate` to guide the entity extraction process.
3. Use the `entity_chain` to extract entities from the input question.
4. Print the extracted entities.

"""

class Entities(BaseModel):
    """Entity information extracted from text."""
    names: List[str] = Field(
        description="List of person, organization, or business entities appearing in the text"
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract organization and person etc entities from the text."),
    ("human", "Extract all the entities from the following input: {question}")
])

entity_chain = prompt | chat.with_structured_output(Entities)

# Test entity extraction
entities = entity_chain.invoke({"question": "How has the company's capital expenditure (CapEx) evolved over the past five years, and what strategic initiatives have driven significant changes in CapEx allocations"})
print(entities.names)

```### Structured Retrieval from Knowledge Graph

DEBUG_MODE = False  # Set to True when you want to see debug info

def structured_retriever(question: str) -> str:
    """
    Retrieves all incoming and outgoing relationships at any depth
    for entities mentioned in the question, matching case-insensitive IDs.
    """
    result = ""
    entities = entity_chain.invoke({"question": question})

    for entity in entities.names:
        if DEBUG_MODE:
            print(f"Getting relationships for entity: {entity}")

        response = kg.query(
            """
            // Find the node matching the entity ID case-insensitively
            MATCH (n)
            WHERE toLower(n.id) = toLower($entity)

            // Outgoing relationships at any depth
            OPTIONAL MATCH pOut = (n)-[*1..]->(m)
            UNWIND relationships(pOut) AS relOut
            WITH DISTINCT relOut AS rel

            RETURN 
              startNode(rel).id + ' -[' + type(rel) + ']-> ' + endNode(rel).id AS output

            UNION

            // Incoming relationships at any depth
            MATCH (n)
            WHERE toLower(n.id) = toLower($entity)
            OPTIONAL MATCH pIn = (m)-[*1..]->(n)
            UNWIND relationships(pIn) AS relIn
            WITH DISTINCT relIn AS rel

            RETURN 
              startNode(rel).id + ' -[' + type(rel) + ']-> ' + endNode(rel).id AS output

            ORDER BY output
            """,
            {"entity": entity}
        )

        if response:
            result += "\n".join(row["output"] for row in response) + "\n"
        else:
            result += f"No relationships found for entity: {entity}\n"

        # Additionally find grandparent IDs (exactly two hops out)
        gp = kg.query(
            """
            MATCH (n)
            WHERE toLower(n.id) = toLower($entity)
            OPTIONAL MATCH (n)-[*2]->(grand)
            RETURN DISTINCT grand.id AS grandparent_id
            LIMIT 1
            """,
            {"entity": entity}
        )
        if gp and gp[0].get("grandparent_id"):
            result += f"Grandparent id of {entity}: {gp[0]['grandparent_id']}\n"
        else:
            result += f"No grandparent found for {entity}\n"

    return result



"""### Retrieval and Chain Processing

This section processes retrieval and chain operations for the Jupyter notebook. The retriever function performs the following steps:

1. Takes a question string as input
2. Performs structured retrieval using the knowledge graph
3. Combine the structured data into a final response

The results are used by the RAG chain to provide comprehensive answers based on both graph relationships and document content.

"""

# Final retrieval step
def retriever(question: str):
    """
    Retrieves information using only the knowledge graph (structured data)
    """
    if DEBUG_MODE:
        print(f"Search query: {question}")
    
    # Get structured data from knowledge graph
    structured_data = structured_retriever(question)
    
    # Return only graph-based data
    final_data = f"""Structured data:
{structured_data}
    """
    
    if DEBUG_MODE:
        print(f"\nFinal Data (Graph Only)::: ==>{final_data}")
    
    return final_data


# Remove the chat history branching - just pass the question through
_search_query = RunnableLambda(lambda x: x["question"])

# Simplified chain without chat history complexity
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | chat
    | StrOutputParser()
)

# Usage (no chat_history needed)
result = chain.invoke({"question": "How has the company's capital expenditure evolved?"})


"""### Response Generation Template and Chain Configuration

This section configures the response generation template and chain for processing queries. It includes:

1. Question-Answer Template:
    - Takes context and question as inputs
    - Structures responses for natural language output
    - Emphasizes concise answers based on provided context

2. Chain Configuration:
    - Combines context retrieval with question processing
    - Uses parallel processing for efficient response generation
    - Integrates chat model for natural language generation


"""

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | chat
    | StrOutputParser()
)

import argparse

DEBUG_MODE = False  # global flag used by structured_retriever/retriever

def run_interactive_chat(verbose=False):
    """Interactive chat with optional verbose mode"""
    global DEBUG_MODE
    DEBUG_MODE = verbose

    print("🤖 Hybrid RAG Chat System")
    if verbose:
        print("🔍 Verbose mode: ON (showing debug information)")
    else:
        print("💡 Clean mode: ON (minimal output)")
    print("Type 'quit' to exit, 'verbose' to toggle debug mode\n")

    while True:
        user_question = input("You: ").strip()
        
        # Exit conditions
        if user_question.lower() in ('quit', 'exit', 'q'):
            print("👋 Goodbye!")
            break
        
        # Toggle verbose at runtime
        if user_question.lower() == 'verbose':
            DEBUG_MODE = not DEBUG_MODE
            print(f"🔄 Verbose mode: {'ON' if DEBUG_MODE else 'OFF'}")
            continue
        
        if not user_question:
            continue
        
        try:
            if not DEBUG_MODE:
                print("🔄 Processing...")
            
            # Invoke your RAG chain
            result = chain.invoke({"question": user_question})
            
            # In verbose mode, show a header
            if DEBUG_MODE:
                print("\n🤖 Final Answer:")
                print("-" * 40)
            
            # Print the answer (and evidence in verbose mode)
            print(result)
            print()
            
            if not DEBUG_MODE:
                print("-" * 30)
            
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG Chat System")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed debug information and evidence",
        default=False
    )
    args = parser.parse_args()
    run_interactive_chat(verbose=args.verbose)

if __name__ == "__main__":
    main()



