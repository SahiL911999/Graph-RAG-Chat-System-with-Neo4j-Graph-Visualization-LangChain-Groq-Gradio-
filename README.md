# 🤖 Hybrid Graph RAG Chat System with Neo4j Visualization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-4.0+-green.svg)](https://neo4j.com/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/)

A sophisticated **Retrieval-Augmented Generation (RAG)** system that combines the power of **Knowledge Graphs** with **Large Language Models** to provide intelligent question-answering with interactive graph visualizations. Built with Neo4j, LangChain, Groq, and Gradio.

![System Architecture](https://img.shields.io/badge/Architecture-Hybrid_RAG-blueviolet)

## 🌟 Features

### Core Capabilities
- **🔍 Hybrid RAG Architecture**: Combines structured graph data with LLM reasoning for accurate, context-aware responses
- **📊 Interactive Knowledge Graph Visualization**: Real-time graph generation using PyVis with Neo4j backend
- **💬 Conversational Interface**: Clean, intuitive Gradio-based chat UI with dual-tab layout
- **🔧 Debug Mode**: Toggle-able verbose logging for development and troubleshooting
- **🚀 Entity Extraction**: Automatic entity recognition from user queries using structured LLM outputs
- **🌐 Multi-depth Relationship Traversal**: Explores incoming and outgoing relationships at any depth in the knowledge graph

### Technical Highlights
- **Parallel Processing**: Efficient document processing with ThreadPoolExecutor
- **Persistent Storage**: Pickle-based caching for processed graph documents
- **Case-Insensitive Matching**: Robust entity matching in Neo4j queries
- **Dynamic Graph Generation**: Automatic HTML graph visualization for each query
- **Browser Integration**: One-click graph opening in new browser tabs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Gradio)                  │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Chat Tab       │         │  Graph Viz Tab   │         │
│  │  - Q&A Interface │         │  - PyVis Network │         │
│  │  - Debug Logs    │         │  - Browser View  │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Processing Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Entity     │→ │  Structured  │→ │   Response   │     │
│  │  Extraction  │  │  Retrieval   │  │  Generation  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Neo4j      │  │   Groq LLM   │  │   CSV Data   │     │
│  │  Knowledge   │  │   (Llama)    │  │   Loader     │     │
│  │    Graph     │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Neo4j Database**: Aura instance or local installation
- **Groq API Key**: For LLM access (Llama 3.3 70B)
- **Environment Variables**: Configured in `.env` file

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone  https://github.com/SahiL911999/Graph-RAG-Chat-System-with-Neo4j-Graph-Visualization-LangChain-Groq-Gradio-.git   
cd Graph-RAG-Chat-System-with-Neo4j-Graph-Visualization-LangChain-Groq-Gradio-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Additional Requirements
```bash
pip install gradio pyvis
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Neo4j Configuration
AURA_INSTANCENAME=your_instance_name
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# API Keys
OPENAI_API_KEY=your_openai_key  # Optional
GROQ_API_KEY=your_groq_api_key
```

## 📊 Data Preparation

### CSV Data Format

The system expects CSV data with the following structure (see [`IY2.csv`](IY2.csv:1)):

```csv
Product,ProductId,ProductType,ProductTypeId,ProductTypeParent,ProductTypeParentId,...
Gmail Toolkit,419547928147927426,Email Software,438671598912938122,...
Youtube,419547928147928131,Streaming Media Software,438671598912938066,...
```

### Processing Pipeline

1. **Document Loading**: CSV files are loaded using [`CSVLoader`](hybrid_graph_rag_structured.py:85)
2. **Graph Transformation**: Documents are converted to graph format using [`LLMGraphTransformer`](hybrid_graph_rag_structured.py:114)
3. **Neo4j Storage**: Graph documents are stored in Neo4j with relationships
4. **Pickle Caching**: Processed documents are cached for faster subsequent runs

## 🎮 Usage

### Starting the Application

#### Method 1: Gradio Web Interface (Recommended)
```bash
python app.py
```
Access the interface at: `http://localhost:7860`

#### Method 2: Command-Line Interface
```bash
# Normal mode
python hybrid_graph_rag_structured.py

# Verbose/Debug mode
python hybrid_graph_rag_structured.py --verbose
```

### Using the Web Interface

1. **Ask Questions**: Type your query in the chat input
2. **View Responses**: Get AI-generated answers based on graph data
3. **Explore Graphs**: Switch to the "Graph Visualization" tab
4. **Open in Browser**: Click "Open Graph in New Tab" for full-screen view
5. **Toggle Debug**: Enable debug mode to see detailed processing logs

### Example Queries

```
"Tell me about Gmail Toolkit"
"What is the relationship between Youtube and Entertainment Software?"
"How has the company's capital expenditure evolved?"
"What products are related to Communication Software?"
```

## 🔧 Configuration

### Debug Mode

Toggle debug mode in the UI or set programmatically:

```python
DEBUG_MODE = True  # In app.py or hybrid_graph_rag_structured.py
```

Debug mode provides:
- Entity extraction details
- Cypher query execution logs
- Node and relationship counts
- Processing timestamps

### Graph Visualization Settings

Customize in [`app.py`](app.py:134):

```python
net = Network(
    height="600px",
    width="100%",
    bgcolor="#ffffff",
    notebook=True
)
```

### LLM Configuration

Modify the model in [`hybrid_graph_rag_structured.py`](hybrid_graph_rag_structured.py:41):

```python
chat = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile"
)
```

## 📁 Project Structure

```
Graph-RAG-Chat-System-with-Neo4j-Graph-Visualization-LangChain-Groq-Gradio-/
│
├── app.py                          # Gradio web interface
├── hybrid_graph_rag_structured.py  # Core RAG logic & CLI
├── requirements.txt                # Python dependencies
├── IY2.csv                        # Sample data
├── graph_documents.pkl            # Cached graph documents
├── .env                           # Environment variables (create this)
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
│
└── Generated HTML Graphs:
    ├── gmail_toolkit_graph.html
    ├── youtube_graph.html
    └── twilio_live_graph.html
```

## 🔑 Key Components

### 1. Entity Extraction ([`hybrid_graph_rag_structured.py`](hybrid_graph_rag_structured.py:239-250))
```python
class Entities(BaseModel):
    names: List[str] = Field(
        description="List of person, organization, or business entities"
    )

entity_chain = prompt | chat.with_structured_output(Entities)
```

### 2. Structured Retrieval ([`hybrid_graph_rag_structured.py`](hybrid_graph_rag_structured.py:260-324))
- Traverses Neo4j graph at any depth
- Retrieves incoming and outgoing relationships
- Case-insensitive entity matching
- Grandparent relationship discovery

### 3. Graph Visualization ([`app.py`](app.py:94-164))
- PyVis network generation
- Dynamic HTML file creation
- Browser integration
- Real-time status updates

### 4. Unified Query Processing ([`app.py`](app.py:166-201))
- Parallel RAG response and graph generation
- Error handling and logging
- Chat history management

## 🎨 UI Features

### Chat Tab
- **Chatbot Interface**: Conversation history with copy functionality
- **Question Input**: Multi-line text input with submit on Enter
- **Action Buttons**: Send, Clear, Toggle Debug
- **Status Display**: Real-time system status and debug mode indicator
- **Debug Panel**: Collapsible detailed logs (when enabled)

### Graph Visualization Tab
- **Status Display**: Graph generation status with node/relationship counts
- **Browser Button**: Opens graph in new tab for full interaction
- **Auto-generation**: Graphs created automatically with each query

## 🔍 Advanced Features

### Multi-depth Relationship Traversal

The system uses sophisticated Cypher queries to explore relationships:

```cypher
MATCH (n) WHERE toLower(n.id)=toLower($entity)
OPTIONAL MATCH pOut=(n)-[*1..]->(m)
UNWIND relationships(pOut) AS relOut
WITH DISTINCT relOut AS rel
RETURN startNode(rel).id + ' -[' + type(rel) + ']-> ' + endNode(rel).id
```

### Parallel Document Processing

Efficient batch processing with progress tracking:

```python
with ThreadPoolExecutor() as executor:
    with tqdm(total=len(documents), desc="Processing Documents") as pbar:
        for i in range(0, len(documents), 100):
            batch = documents[i : i + 100]
            # Process batch in parallel
```

### Persistent Caching

Automatic pickle-based caching to avoid reprocessing:

```python
if os.path.exists(PICKLE_PATH):
    with open(PICKLE_PATH, "rb") as f:
        graph_documents = pickle.load(f)
```

## 🐛 Troubleshooting

### Common Issues

**1. Neo4j Connection Error**
```
Solution: Verify NEO4J_URI, username, and password in .env
Check if Neo4j instance is running
```

**2. No Graph Data Found**
```
Solution: Ensure CSV data is loaded and processed
Run the document processing pipeline first
Check if entities exist in Neo4j database
```

**3. Gradio Port Already in Use**
```
Solution: Change port in app.py:
demo.launch(server_port=7861)  # Use different port
```

**4. Missing Dependencies**
```
Solution: Install all requirements:
pip install -r requirements.txt
pip install gradio pyvis
```

## 🚦 Performance Optimization

- **Batch Processing**: Documents processed in batches of 100
- **Pickle Caching**: Avoids reprocessing on subsequent runs
- **Parallel Execution**: ThreadPoolExecutor for concurrent processing
- **Database Indexing**: Neo4j indexes on entity IDs for fast lookups

## 📈 Future Enhancements

- [ ] Support for PDF and DOCX document loading
- [ ] Multi-language support
- [ ] Advanced graph filtering and search
- [ ] Export functionality for graphs and conversations
- [ ] Integration with additional LLM providers
- [ ] Real-time collaborative features
- [ ] Custom entity type definitions
- [ ] Graph analytics and insights dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE:1) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the powerful RAG framework
- **Neo4j**: For the graph database platform
- **Groq**: For fast LLM inference
- **Gradio**: For the intuitive UI framework
- **PyVis**: For interactive network visualizations

## 👨‍💻 Contributors

- **Sahil Rannmbail** - *Initial work and development*

## 📞 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review the debug logs with verbose mode enabled

## 🌐 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Neo4j Graph Database](https://neo4j.com/docs/)
- [Gradio Documentation](https://gradio.app/docs/)
- [Groq API](https://groq.com/)

---

**Built with ❤️ using Graph RAG Technology**

*Last Updated: 2025*