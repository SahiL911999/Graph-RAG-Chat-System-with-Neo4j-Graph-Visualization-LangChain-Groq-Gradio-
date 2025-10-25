import gradio as gr
from neo4j import GraphDatabase
from pyvis.network import Network
import os
import webbrowser
from dotenv import load_dotenv
from hybrid_graph_rag_structured import entity_chain, chain

load_dotenv()

# Global flag for debug mode
DEBUG_MODE = False
debug_logs = []

def log_debug(message):
    """Add message to debug logs"""
    global debug_logs
    if DEBUG_MODE:
        debug_logs.append(message)
        print(message)  # Also print to console

def get_debug_logs():
    """Return formatted debug logs"""
    global debug_logs
    if not debug_logs:
        return ""
    return "\n".join(debug_logs)

def clear_debug_logs():
    """Clear debug logs"""
    global debug_logs
    debug_logs = []

def get_graph_data(tx, entity):
    """Fetch graph relationships for a given entity - exact logic from working code"""
    query = f"""
    MATCH (n) WHERE toLower(n.id)=toLower('{entity}')
    OPTIONAL MATCH pOut=(n)-[*1..]->(m)
    UNWIND relationships(pOut) AS relOut
    WITH DISTINCT relOut AS rel
    RETURN startNode(rel) AS source, type(rel) AS rel_type, endNode(rel) AS target
    UNION
    MATCH (n) WHERE toLower(n.id)=toLower('{entity}')
    OPTIONAL MATCH pIn=(m)-[*1..]->(n)
    UNWIND relationships(pIn) AS relIn
    WITH DISTINCT relIn AS rel
    RETURN startNode(rel) AS source, type(rel) AS rel_type, endNode(rel) AS target
    """
    
    log_debug(f"\n{'='*60}")
    log_debug(f"🔍 EXECUTING CYPHER QUERY FOR ENTITY: {entity}")
    log_debug(f"{'='*60}")
    log_debug(f"\n{query}\n")
    log_debug(f"{'='*60}\n")
    
    result = tx.run(query)
    node_map = {}
    relationships = []
    idx = 0
    
    log_debug("📊 PROCESSING QUERY RESULTS:")
    log_debug("-" * 60)
    
    for record in result:
        s, t, r = record["source"], record["target"], record["rel_type"]
        s_id = s.get("id", s.element_id)
        t_id = t.get("id", t.element_id)
        
        if s_id not in node_map:
            node_map[s_id] = idx
            log_debug(f"➕ New Node: {s_id} (ID: {idx})")
            idx += 1
        if t_id not in node_map:
            node_map[t_id] = idx
            log_debug(f"➕ New Node: {t_id} (ID: {idx})")
            idx += 1
            
        relationships.append({
            "source": node_map[s_id],
            "target": node_map[t_id],
            "caption": r
        })
        
        log_debug(f"🔗 Relationship: {s_id} -[{r}]-> {t_id}")
    
    log_debug(f"\n{'='*60}")
    log_debug(f"✅ QUERY COMPLETE:")
    log_debug(f"   Total Nodes: {len(node_map)}")
    log_debug(f"   Total Relationships: {len(relationships)}")
    log_debug(f"{'='*60}\n")
    
    return node_map, relationships

def generate_graph_visualization(question):
    """Generate graph visualization from question - using exact working logic"""
    try:
        log_debug(f"\n{'#'*60}")
        log_debug(f"🎯 STARTING GRAPH GENERATION")
        log_debug(f"{'#'*60}")
        log_debug(f"📝 Question: {question}\n")
        
        # Extract entities from question
        log_debug("🔎 EXTRACTING ENTITIES...")
        ents = entity_chain.invoke({"question": question})
        
        log_debug(f"✅ Entities Found: {ents.names if ents.names else ['None - using default']}")
        
        entity = ents.names[0] if ents.names else "Youtube"
        log_debug(f"🎯 Primary Entity Selected: {entity}\n")
        
        # Connect to Neo4j and get data
        log_debug("🔌 CONNECTING TO NEO4J DATABASE...")
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD")
        
        log_debug(f"   URI: {uri}")
        log_debug(f"   User: {user}\n")
        
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        
        with driver.session() as session:
            node_map, relationships = session.execute_read(get_graph_data, entity)
        
        driver.close()
        log_debug("✅ Database connection closed\n")
        
        if not node_map:
            log_debug(f"❌ No graph data found for entity: {entity}")
            return None, f"❌ No graph data found for entity: {entity}", gr.update(visible=False)
        
        # Create PyVis network
        log_debug("🎨 CREATING PYVIS NETWORK VISUALIZATION...")
        net = Network(height="600px", width="100%", bgcolor="#ffffff", notebook=True)
        
        # Add nodes
        log_debug("➕ Adding nodes to visualization:")
        for name, i in sorted(node_map.items(), key=lambda x: x[1]):
            net.add_node(i, label=name, title=f"Node ID: {i}")
            log_debug(f"   Node {i}: {name}")
        
        # Add edges
        log_debug("\n🔗 Adding edges to visualization:")
        for rel in relationships:
            net.add_edge(rel["source"], rel["target"], title=rel["caption"])
            log_debug(f"   Edge: {rel['source']} -[{rel['caption']}]-> {rel['target']}")
        
        # Save HTML file
        filename = f"{entity.lower().replace(' ', '_')}_graph.html"
        net.write_html(filename)
        
        log_debug(f"\n💾 Graph saved to: {filename}")
        log_debug(f"\n{'#'*60}")
        log_debug(f"✅ GRAPH GENERATION COMPLETE")
        log_debug(f"{'#'*60}\n")
        
        status_msg = f"✅ Graph generated for entity: **{entity}** | Nodes: {len(node_map)} | Relationships: {len(relationships)}"
        
        return filename, status_msg, gr.update(visible=True)
        
    except Exception as e:
        error_msg = f"❌ Error generating graph: {str(e)}"
        log_debug(f"\n{error_msg}")
        return None, error_msg, gr.update(visible=False)

def process_unified_query(question, chat_history):
    """Process query - get chat answer AND generate graph in one go"""
    
    # Clear previous debug logs
    clear_debug_logs()
    
    if not question.strip():
        return chat_history, "", None, "⚠️ Please enter a question", gr.update(visible=False), ""
    
    log_debug(f"\n{'*'*60}")
    log_debug(f"🚀 NEW QUERY PROCESSING STARTED")
    log_debug(f"{'*'*60}")
    log_debug(f"📝 User Question: {question}\n")
    
    # Step 1: Get RAG answer for chat
    try:
        log_debug("💬 GENERATING RAG RESPONSE...")
        log_debug("-" * 60)
        result = chain.invoke({"question": question})
        log_debug(f"✅ RAG Response Generated:")
        log_debug(f"{result[:200]}..." if len(result) > 200 else result)
        log_debug("-" * 60 + "\n")
        chat_history.append((question, result))
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        log_debug(f"\n❌ RAG ERROR: {error_msg}\n")
        chat_history.append((question, error_msg))
        return chat_history, "", None, error_msg, gr.update(visible=False), get_debug_logs()
    
    # Step 2: Generate graph visualization
    filename, graph_status, button_update = generate_graph_visualization(question)
    
    # Return debug logs
    debug_output = get_debug_logs() if DEBUG_MODE else ""
    
    return chat_history, "", filename, graph_status, button_update, debug_output

def open_graph_in_browser(filename):
    """Open the generated graph HTML file in browser"""
    if filename:
        try:
            path = os.path.abspath(filename)
            webbrowser.open(f'file://{path}')
            log_debug(f"🌐 Opening graph in browser: {path}")
            return "🌐 Graph opened in new browser tab!"
        except Exception as e:
            error_msg = f"❌ Error opening graph: {str(e)}"
            log_debug(error_msg)
            return error_msg
    return "❌ No graph file to open"

def toggle_debug_mode(current_status):
    """Toggle debug mode on/off"""
    global DEBUG_MODE
    DEBUG_MODE = not DEBUG_MODE
    
    # Update debug mode in the structured retriever module
    try:
        import hybrid_graph_rag_structured
        hybrid_graph_rag_structured.DEBUG_MODE = DEBUG_MODE
    except:
        pass
    
    status = "🔍 Debug Mode: ON" if DEBUG_MODE else "💡 Debug Mode: OFF"
    
    # Show/hide debug panel
    debug_visibility = gr.update(visible=DEBUG_MODE)
    
    return status, debug_visibility

def clear_all():
    """Clear chat history, graph, and debug logs"""
    clear_debug_logs()
    return [], "", None, "Ready to process your question...", gr.update(visible=False), ""

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Hybrid RAG Chat with Graph Visualization") as demo:
    
    gr.Markdown("""
    # 🤖 Hybrid RAG Chat System with Graph Visualization
    
    Ask a question to get both a text answer and an interactive knowledge graph visualization.
    """)
    
    # Shared state for graph filename
    graph_filename = gr.State()
    
    with gr.Tabs():
        # Tab 1: Chat Interface
        with gr.Tab("💬 Chat"):
            
            chatbot = gr.Chatbot(
                label="Chat History",
                height=450,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your question here...",
                    lines=2,
                    scale=4
                )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Send", variant="primary", scale=2, size="lg")
                clear_btn = gr.Button("🗑️ Clear", scale=1)
                debug_btn = gr.Button("🔧 Toggle Debug", scale=1)
            
            debug_status = gr.Textbox(
                label="System Status",
                value="💡 Debug Mode: OFF",
                interactive=False
            )
            
            # Debug panel - hidden by default
            debug_panel = gr.Textbox(
                label="🔍 Debug Logs",
                lines=15,
                max_lines=20,
                visible=False,
                interactive=False,
                show_copy_button=True
            )
        
        # Tab 2: Graph Visualization
        with gr.Tab("📊 Graph Visualization"):
            gr.Markdown("""
            ### Interactive Knowledge Graph
            
            The graph is automatically generated when you ask a question in the Chat tab.
            """)
            
            graph_status_display = gr.Markdown("Ready to process your question...")
            
            open_graph_btn = gr.Button(
                "🌐 Open Graph in New Tab",
                variant="primary",
                visible=False,
                size="lg",
                scale=1
            )
    
    # Event handlers - unified processing
    submit_btn.click(
        fn=process_unified_query,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, graph_filename, graph_status_display, open_graph_btn, debug_panel]
    )
    
    question_input.submit(
        fn=process_unified_query,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, graph_filename, graph_status_display, open_graph_btn, debug_panel]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot, question_input, graph_filename, graph_status_display, open_graph_btn, debug_panel]
    )
    
    debug_btn.click(
        fn=toggle_debug_mode,
        inputs=[debug_status],
        outputs=[debug_status, debug_panel]
    )
    
    open_graph_btn.click(
        fn=open_graph_in_browser,
        inputs=graph_filename,
        outputs=graph_status_display
    )

if __name__ == "__main__":
    print("🚀 Starting Unified Hybrid RAG Chat System...")
    print("💡 One question → Text answer + Graph visualization")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )