import os
import json
import uuid
import re
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO
from dotenv import load_dotenv
import threading
import time
from argparse import Namespace

from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs,
    RunnerArgument,
    CoStormRunner,
)
from knowledge_storm.collaborative_storm.modules.callback import (
    LocalConsolePrintCallBackHandler,
)
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import (
    BingSearch,
    SerperRM,
    TavilySearchRM,
)
from knowledge_storm.utils import load_api_key

# Load environment variables
load_dotenv()
load_api_key(toml_file_path="secrets.toml")

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active research sessions
active_sessions = {}

# Constants
NUM_TURNS = 10  # Number of conversation turns
HUMAN_WAIT_TIME = int(os.getenv("HUMAN_WAIT_TIME", "10"))  # Seconds to wait for human input

def create_costorm_runner(topic):
    """Create and initialize a CoStormRunner instance."""
    lm_config = CollaborativeStormLMConfigs()
    
    # Configure OpenAI settings
    openai_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_provider": "openai",
        "temperature": 1.0,
        "top_p": 0.9,
        "api_base": None,
    }
    
    # Use GPT-4o models
    model_name = "gpt-4o"
    
    # Initialize models for different components
    question_answering_lm = OpenAIModel(
        model=model_name, max_tokens=1000, **openai_kwargs
    )
    discourse_manage_lm = OpenAIModel(
        model=model_name, max_tokens=500, **openai_kwargs
    )
    utterance_polishing_lm = OpenAIModel(
        model=model_name, max_tokens=2000, **openai_kwargs
    )
    warmstart_outline_gen_lm = OpenAIModel(
        model=model_name, max_tokens=500, **openai_kwargs
    )
    question_asking_lm = OpenAIModel(
        model=model_name, max_tokens=300, **openai_kwargs
    )
    knowledge_base_lm = OpenAIModel(
        model=model_name, max_tokens=1000, **openai_kwargs
    )
    
    # Set models in config
    lm_config.set_question_answering_lm(question_answering_lm)
    lm_config.set_discourse_manage_lm(discourse_manage_lm)
    lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
    lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
    lm_config.set_question_asking_lm(question_asking_lm)
    lm_config.set_knowledge_base_lm(knowledge_base_lm)
    
    # Configure runner arguments
    runner_argument = RunnerArgument(
        topic=topic,
        retrieve_top_k=10,
        max_search_queries=2,
        total_conv_turn=NUM_TURNS,
        max_search_thread=5,
        max_search_queries_per_turn=3,
        warmstart_max_num_experts=3,
        warmstart_max_turn_per_experts=2,
        warmstart_max_thread=3,
        max_thread_num=10,
        max_num_round_table_experts=2,
        moderator_override_N_consecutive_answering_turn=3,
        node_expansion_trigger_count=10,
    )
    
    # Initialize logging
    logging_wrapper = LoggingWrapper(lm_config)
    
    # Initialize retrieval module (using Serper by default)
    retriever_type = os.getenv("RETRIEVER_TYPE", "serper")
    
    if retriever_type == "bing":
        rm = BingSearch(
            bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
            k=runner_argument.retrieve_top_k,
        )
    elif retriever_type == "tavily":
        rm = TavilySearchRM(
            tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
            k=runner_argument.retrieve_top_k,
            include_raw_content=True,
        )
    else:  # Default to serper
        rm = SerperRM(
            serper_search_api_key=os.getenv("SERPER_API_KEY"),
            query_params={"autocorrect": True, "num": 10, "page": 1},
        )
    
    # Create and return the CoStormRunner
    costorm_runner = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=rm,
        callback_handler=None,  # We'll set this in run_costorm_session
    )
    
    return costorm_runner

def process_citations(article, instance_dump):
    """Process citations in the article and add a references section."""
    print("Starting citation processing...")
    
    # Extract citation information from instance_dump
    citations = {}
    
    # Look for citations in knowledge_base.info_uuid_to_info_dict
    if "knowledge_base" in instance_dump and "info_uuid_to_info_dict" in instance_dump["knowledge_base"]:
        info_dict = instance_dump["knowledge_base"]["info_uuid_to_info_dict"]
        print(f"Found info_uuid_to_info_dict with {len(info_dict)} entries")
        
        for citation_id, citation_info in info_dict.items():
            # Convert citation_id to string if it's an integer
            citation_id_str = str(citation_id)
            if "url" in citation_info and "title" in citation_info:
                citations[citation_id_str] = {
                    "url": citation_info["url"],
                    "title": citation_info["title"]
                }
                print(f"Added citation {citation_id_str}: {citation_info['title']}")
    else:
        print("No info_uuid_to_info_dict found in knowledge_base")
    
    print(f"Total citations found: {len(citations)}")
    
    # Create HTML version with clickable links
    html_article = article
    
    # Replace citation markers with links in HTML version
    def replace_citation_html(match):
        citation_id = match.group(1)
        # Try both the original citation_id and as a string
        if citation_id in citations:
            print(f"Replacing citation [{citation_id}] with link to {citations[citation_id]['url']}")
            return f'<a href="{citations[citation_id]["url"]}" target="_blank" class="citation-link">[{citation_id}]</a>'
        # Try without leading zeros if it's a number
        elif citation_id.lstrip('0') in citations:
            clean_id = citation_id.lstrip('0')
            print(f"Replacing citation [{citation_id}] with link to {citations[clean_id]['url']} (cleaned ID: {clean_id})")
            return f'<a href="{citations[clean_id]["url"]}" target="_blank" class="citation-link">[{citation_id}]</a>'
        print(f"Citation [{citation_id}] not found in citations dictionary")
        return match.group(0)
    
    # Count citations in the article
    citation_matches = re.findall(r'\[(\d+)\]', html_article)
    print(f"Found {len(citation_matches)} citation markers in the article")
    for match in citation_matches[:10]:  # Print first 10 for debugging
        print(f"Citation marker: [{match}]")
    
    html_article = re.sub(r'\[(\d+)\]', replace_citation_html, html_article)
    
    # Add references section to both versions
    if citations:
        print("Adding references section")
        # HTML version with clickable links
        html_references = "\n\n# References\n\n"
        for citation_id in sorted(citations.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else float('inf')):
            html_references += f'{citation_id}. <a href="{citations[citation_id]["url"]}" target="_blank" rel="noopener noreferrer" class="citation-link">{citations[citation_id]["title"]}</a>\n'
        html_article += html_references
        
        # Plain markdown version with URLs
        markdown_references = "\n\n# References\n\n"
        for citation_id in sorted(citations.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else float('inf')):
            markdown_references += f'{citation_id}. [{citations[citation_id]["title"]}]({citations[citation_id]["url"]})\n'
        article += markdown_references
    else:
        print("No citations found, skipping references section")
    
    # Ensure proper Markdown formatting for headers
    # This will help the frontend render the Markdown correctly
    html_article = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_article, flags=re.MULTILINE)
    html_article = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_article, flags=re.MULTILINE)
    html_article = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_article, flags=re.MULTILINE)
    
    # Convert paragraphs to proper HTML
    paragraphs = html_article.split('\n\n')
    processed_paragraphs = []
    for paragraph in paragraphs:
        # Skip if it's already an HTML tag or empty
        if paragraph.strip() == '' or paragraph.strip().startswith('<'):
            processed_paragraphs.append(paragraph)
        else:
            # Wrap in paragraph tags
            processed_paragraphs.append(f'<p>{paragraph}</p>')
    
    html_article = '\n\n'.join(processed_paragraphs)
    
    print("Citation processing complete")
    return {
        "html": html_article,
        "markdown": article
    }

def debug_instance_dump(instance_dump, max_depth=2, current_depth=0, path=""):
    """Debug function to print the structure of the instance_dump.json file."""
    if current_depth > max_depth:
        return
    
    indent = "  " * current_depth
    
    if isinstance(instance_dump, dict):
        print(f"{indent}Dict at {path} with {len(instance_dump)} keys: {list(instance_dump.keys())[:10]}")
        if current_depth < max_depth:
            for key, value in list(instance_dump.items())[:5]:  # Limit to first 5 items
                new_path = f"{path}.{key}" if path else key
                print(f"{indent}Key: {key}")
                debug_instance_dump(value, max_depth, current_depth + 1, new_path)
    elif isinstance(instance_dump, list):
        print(f"{indent}List at {path} with {len(instance_dump)} items")
        if current_depth < max_depth and len(instance_dump) > 0:
            for i, item in enumerate(instance_dump[:3]):  # Limit to first 3 items
                new_path = f"{path}[{i}]"
                debug_instance_dump(item, max_depth, current_depth + 1, new_path)
    else:
        print(f"{indent}Value at {path}: {type(instance_dump).__name__}")

def run_costorm_session(session_id, topic):
    """Run the Co-Storm algorithm in a separate thread."""
    try:
        # Create output directory for this session
        output_dir = f"./results/{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the runner with custom callback handler
        callback_handler = UIStatusCallbackHandler(socketio, session_id)
        costorm_runner = create_costorm_runner(topic)
        active_sessions[session_id]["runner"] = costorm_runner
        
        # Set the callback handler
        costorm_runner.callback_handler = callback_handler
        
        # Warm start the system - use only status updates, not message
        socketio.emit('status_update', {'status': 'Starting warm-up phase...'}, room=session_id)
        costorm_runner.warm_start()
        
        # Run the conversation for NUM_TURNS turns
        turn_count = 0
        while turn_count < NUM_TURNS and not active_sessions[session_id]["completed"]:
            # Check if there's a user message to process
            if active_sessions[session_id]["user_message"]:
                user_message = active_sessions[session_id]["user_message"]
                active_sessions[session_id]["user_message"] = None
                
                # Process user message
                costorm_runner.step(user_utterance=user_message)
                
                # Get AI response
                conv_turn = costorm_runner.step()
                
                # Send message to client
                socketio.emit('message', {
                    'role': conv_turn.role,
                    'content': conv_turn.utterance
                }, room=session_id)
                
                turn_count += 1
                
                # Set waiting for human input flag
                active_sessions[session_id]["waiting_for_human"] = True
                
                # Wait for human input or timeout
                wait_start_time = time.time()
                while (time.time() - wait_start_time < HUMAN_WAIT_TIME and 
                       active_sessions[session_id]["waiting_for_human"] and 
                       not active_sessions[session_id]["user_message"] and
                       not active_sessions[session_id]["user_typing"] and
                       not active_sessions[session_id]["completed"]):
                    time.sleep(0.5)
                
                # If user is typing, wait until they're done or submit a message
                while (active_sessions[session_id]["user_typing"] and 
                       not active_sessions[session_id]["user_message"] and
                       not active_sessions[session_id]["completed"]):
                    time.sleep(0.5)
                
                # Reset waiting flag
                active_sessions[session_id]["waiting_for_human"] = False
                
                # If user provided a message, process it in the next loop iteration
                if active_sessions[session_id]["user_message"]:
                    continue
                
            else:
                # If no user message, get AI response
                conv_turn = costorm_runner.step()
                
                # Send message to client
                socketio.emit('message', {
                    'role': conv_turn.role,
                    'content': conv_turn.utterance
                }, room=session_id)
                
                turn_count += 1
                
                # Set waiting for human input flag
                active_sessions[session_id]["waiting_for_human"] = True
                
                # Wait for human input or timeout
                wait_start_time = time.time()
                while (time.time() - wait_start_time < HUMAN_WAIT_TIME and 
                       active_sessions[session_id]["waiting_for_human"] and 
                       not active_sessions[session_id]["user_message"] and
                       not active_sessions[session_id]["user_typing"] and
                       not active_sessions[session_id]["completed"]):
                    time.sleep(0.5)
                
                # If user is typing, wait until they're done or submit a message
                while (active_sessions[session_id]["user_typing"] and 
                       not active_sessions[session_id]["user_message"] and
                       not active_sessions[session_id]["completed"]):
                    time.sleep(0.5)
                
                # Reset waiting flag
                active_sessions[session_id]["waiting_for_human"] = False
                
                # If user provided a message, process it in the next loop iteration
                if active_sessions[session_id]["user_message"]:
                    continue
            
            # Small delay to prevent overwhelming the client
            time.sleep(0.5)
        
        # Generate final report
        socketio.emit('message', {'role': 'system', 'content': 'Generating final research article...'}, room=session_id)
        costorm_runner.knowledge_base.reogranize()
        article = costorm_runner.generate_report()
        
        # Save the original article to file
        with open(os.path.join(output_dir, "original_report.md"), "w") as f:
            f.write(article)
        
        # Save instance dump with the correct filename
        instance_copy = costorm_runner.to_dict()
        with open(os.path.join(output_dir, "costorm_instance_dump.json"), "w") as f:
            json.dump(instance_copy, f, indent=2)
        
        # Also save with the standard filename for compatibility
        with open(os.path.join(output_dir, "instance_dump.json"), "w") as f:
            json.dump(instance_copy, f, indent=2)
        
        # Debug: Print instance_dump structure
        print("\n\n===== INSTANCE DUMP STRUCTURE =====")
        debug_instance_dump(instance_copy, max_depth=3)
        print("===== END INSTANCE DUMP STRUCTURE =====\n\n")
        
        # Save logging
        log_dump = costorm_runner.dump_logging_and_reset()
        with open(os.path.join(output_dir, "log.json"), "w") as f:
            json.dump(log_dump, f, indent=2)
        
        # Process citations and add references section
        print("\n\n===== PROCESSING CITATIONS =====")
        # Check if info_uuid_to_info_dict exists in the knowledge_base
        if "knowledge_base" in instance_copy:
            print("Knowledge base found")
            if "info_uuid_to_info_dict" in instance_copy["knowledge_base"]:
                info_dict = instance_copy["knowledge_base"]["info_uuid_to_info_dict"]
                print(f"info_uuid_to_info_dict found with {len(info_dict)} entries")
                # Print a few sample entries
                for key in list(info_dict.keys())[:3]:
                    print(f"Sample citation {key}: {info_dict[key].get('title', 'No title')} - {info_dict[key].get('url', 'No URL')}")
            else:
                print("info_uuid_to_info_dict not found in knowledge_base")
                print(f"Available keys in knowledge_base: {list(instance_copy['knowledge_base'].keys())}")
        else:
            print("Knowledge base not found in instance_copy")
        
        processed_article = process_citations(article, instance_copy)
        print("===== CITATION PROCESSING COMPLETE =====\n\n")
        
        # Save the markdown version to file
        with open(os.path.join(output_dir, "report.md"), "w") as f:
            f.write(processed_article["markdown"])
        
        # Send final article to client
        socketio.emit('final_article', {
            'content': processed_article["html"],
            'markdown': processed_article["markdown"]
        }, room=session_id)
        
        # Mark session as completed
        active_sessions[session_id]["completed"] = True
        
    except Exception as e:
        socketio.emit('error', {'message': f"Error: {str(e)}"}, room=session_id)
        print(f"Error in session {session_id}: {str(e)}")
        import traceback
        traceback.print_exc()

# Custom callback handler to update UI status
class UIStatusCallbackHandler(LocalConsolePrintCallBackHandler):
    """Callback handler that updates the UI status."""
    
    def __init__(self, socketio, session_id):
        super().__init__()
        self.socketio = socketio
        self.session_id = session_id
    
    def _send_status_update(self, status):
        """Send status update to the UI."""
        self.socketio.emit('status_update', {'status': status}, room=self.session_id)
        print(f"Status update: {status}")
    
    def on_turn_policy_planning_start(self, **kwargs):
        """Run when the turn policy planning begins."""
        self._send_status_update("Planning next conversation turn...")
        super().on_turn_policy_planning_start(**kwargs)
    
    def on_expert_action_planning_start(self, **kwargs):
        """Run when the expert action planning begins."""
        self._send_status_update("Planning expert actions...")
        super().on_expert_action_planning_start(**kwargs)
    
    def on_expert_action_planning_end(self, **kwargs):
        """Run when the expert action planning ends."""
        self._send_status_update("Expert actions planned")
        super().on_expert_action_planning_end(**kwargs)
    
    def on_expert_information_collection_start(self, **kwargs):
        """Run when the expert information collection starts."""
        self._send_status_update("Collecting information...")
        super().on_expert_information_collection_start(**kwargs)
    
    def on_expert_information_collection_end(self, info, **kwargs):
        """Run when the expert information collection ends."""
        self._send_status_update(f"Collected {len(info)} pieces of information")
        super().on_expert_information_collection_end(info, **kwargs)
    
    def on_expert_utterance_generation_end(self, **kwargs):
        """Run when the expert utterance generation ends."""
        self._send_status_update("Generated expert response")
        super().on_expert_utterance_generation_end(**kwargs)
    
    def on_expert_utterance_polishing_start(self, **kwargs):
        """Run when the expert utterance polishing begins."""
        self._send_status_update("Refining expert response...")
        super().on_expert_utterance_polishing_start(**kwargs)
    
    def on_mindmap_insert_start(self, **kwargs):
        """Run when the process of inserting new information into the mindmap starts."""
        self._send_status_update("Organizing information...")
        super().on_mindmap_insert_start(**kwargs)
    
    def on_mindmap_insert_end(self, **kwargs):
        """Run when the process of inserting new information into the mindmap ends."""
        self._send_status_update("Information organized")
        super().on_mindmap_insert_end(**kwargs)
    
    def on_mindmap_reorg_start(self, **kwargs):
        """Run when the reorganization of the mindmap begins."""
        self._send_status_update("Reorganizing knowledge structure...")
        super().on_mindmap_reorg_start(**kwargs)
    
    def on_expert_list_update_start(self, **kwargs):
        """Run when the expert list update starts."""
        self._send_status_update("Updating expert list...")
        super().on_expert_list_update_start(**kwargs)
    
    def on_article_generation_start(self, **kwargs):
        """Run when the article generation process begins."""
        self._send_status_update("Generating final article...")
        super().on_article_generation_start(**kwargs)
    
    def on_warmstart_update(self, message, **kwargs):
        """Run when the warm start process has update."""
        self._send_status_update(f"Warming up: {message}")
        super().on_warmstart_update(message, **kwargs)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    session_id = request.sid
    print(f"Client connected: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    session_id = request.sid
    print(f"Client disconnected: {session_id}")
    
    # Clean up session if it exists
    if session_id in active_sessions:
        active_sessions[session_id]["completed"] = True
        del active_sessions[session_id]

@socketio.on('start_research')
def handle_start_research(data):
    """Start a new research session."""
    session_id = request.sid
    topic = data.get('topic', '')
    
    if not topic:
        socketio.emit('error', {'message': 'Please provide a research topic'}, room=session_id)
        return
    
    # Initialize session data
    active_sessions[session_id] = {
        "topic": topic,
        "runner": None,
        "user_message": None,
        "user_typing": False,
        "waiting_for_human": False,
        "completed": False
    }
    
    # Start research in a separate thread
    thread = threading.Thread(target=run_costorm_session, args=(session_id, topic))
    thread.daemon = True
    thread.start()
    
    socketio.emit('research_started', {'topic': topic}, room=session_id)

@socketio.on('send_message')
def handle_send_message(data):
    """Handle user message."""
    session_id = request.sid
    message = data.get('message', '')
    
    if not message:
        return
    
    if session_id in active_sessions and not active_sessions[session_id]["completed"]:
        # Store user message to be processed by the research thread
        active_sessions[session_id]["user_message"] = message
        active_sessions[session_id]["user_typing"] = False
        
        # Echo user message back to client
        socketio.emit('message', {'role': 'user', 'content': message}, room=session_id)
    else:
        socketio.emit('error', {'message': 'No active research session'}, room=session_id)

@socketio.on('typing_started')
def handle_typing_started():
    """Handle user typing notification."""
    session_id = request.sid
    
    if session_id in active_sessions and not active_sessions[session_id]["completed"]:
        active_sessions[session_id]["user_typing"] = True

@socketio.on('typing_stopped')
def handle_typing_stopped():
    """Handle user stopped typing notification."""
    session_id = request.sid
    
    if session_id in active_sessions and not active_sessions[session_id]["completed"]:
        active_sessions[session_id]["user_typing"] = False

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001) 