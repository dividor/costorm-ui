import os
import json
import uuid
import re
import threading
import time
from argparse import Namespace
from collections import defaultdict

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO
from dotenv import load_dotenv

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

app = Flask(__name__)
# Use a persistent secret key from environment variables or generate a random one
# In production, SECRET_KEY should be set in the environment
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))
# If a random key was generated, warn that sessions will be lost on restart
if not os.getenv('SECRET_KEY'):
    print("WARNING: Using a random secret key. Sessions will be lost when the server restarts.")
    print("Set the SECRET_KEY environment variable for persistent sessions.")

# Configure CORS for Socket.IO
# In production, specify allowed origins instead of "*"
cors_origins = os.getenv('CORS_ORIGINS', '*')
socketio = SocketIO(app, cors_allowed_origins=cors_origins)
if cors_origins == '*':
    print("WARNING: CORS is configured to allow all origins. This is not recommended for production.")
    print("Set the CORS_ORIGINS environment variable to restrict allowed origins.")

# Store active research sessions
active_sessions = {}

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # 60 seconds (1 minute)
MAX_REQUESTS_PER_WINDOW = 30  # 30 requests per minute
rate_limit_data = defaultdict(list)

def is_rate_limited(session_id):
    """Check if a session is currently rate limited.
    
    Args:
        session_id (str): The session ID to check.
        
    Returns:
        bool: True if the session is rate limited, False otherwise.
    """
    current_time = time.time()
    
    # Remove timestamps older than the window
    rate_limit_data[session_id] = [
        timestamp for timestamp in rate_limit_data[session_id]
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check if the number of requests in the window exceeds the limit
    if len(rate_limit_data[session_id]) >= MAX_REQUESTS_PER_WINDOW:
        return True
    
    # Add the current timestamp to the list
    rate_limit_data[session_id].append(current_time)
    return False

# Application constants
NUM_TURNS = int(os.getenv("NUM_TURNS", 10))  # Number of conversation turns
HUMAN_WAIT_TIME = int(os.getenv("HUMAN_WAIT_TIME", 10))  # Seconds to wait for human input
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 10))
MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", 2))
MAX_SEARCH_THREAD = int(os.getenv("MAX_SEARCH_THREAD", 5))
MAX_SEARCH_QUERIES_PER_TURN = int(os.getenv("MAX_SEARCH_QUERIES_PER_TURN", 3))
WARMSTART_MAX_NUM_EXPERTS = int(os.getenv("WARMSTART_MAX_NUM_EXPERTS", 3))
WARMSTART_MAX_TURN_PER_EXPERTS = int(os.getenv("WARMSTART_MAX_TURN_PER_EXPERTS", 2))
WARMSTART_MAX_THREAD = int(os.getenv("WARMSTART_MAX_THREAD", 3))
MAX_THREAD_NUM = int(os.getenv("MAX_THREAD_NUM", 10))
MAX_NUM_ROUND_TABLE_EXPERTS = int(os.getenv("MAX_NUM_ROUND_TABLE_EXPERTS", 2))
MODERATOR_OVERRIDE_N_CONSECUTIVE_ANSWERING_TURN = int(os.getenv("MODERATOR_OVERRIDE_N_CONSECUTIVE_ANSWERING_TURN", 3))
NODE_EXPANSION_TRIGGER_COUNT = int(os.getenv("NODE_EXPANSION_TRIGGER_COUNT", 10))

def create_costorm_runner(topic):
    """Create and initialize a CoStormRunner instance.
    
    This function sets up a CoStormRunner with appropriate language models,
    retrieval modules, and configuration settings for the Co-Storm algorithm.
    
    Args:
        topic (str): The research topic to be explored by the Co-Storm algorithm.
        
    Returns:
        CoStormRunner: A fully configured CoStormRunner instance ready to execute
            the Co-Storm algorithm on the specified topic.
            
    Note:
        The function uses environment variables for API keys and configuration settings.
        Make sure these are properly set in the .env file before running.
    """
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
        retrieve_top_k=RETRIEVE_TOP_K,
        max_search_queries=MAX_SEARCH_QUERIES,
        total_conv_turn=NUM_TURNS,
        max_search_thread=MAX_SEARCH_THREAD,
        max_search_queries_per_turn=MAX_SEARCH_QUERIES_PER_TURN,
        warmstart_max_num_experts=WARMSTART_MAX_NUM_EXPERTS,
        warmstart_max_turn_per_experts=WARMSTART_MAX_TURN_PER_EXPERTS,
        warmstart_max_thread=WARMSTART_MAX_THREAD,
        max_thread_num=MAX_THREAD_NUM,
        max_num_round_table_experts=MAX_NUM_ROUND_TABLE_EXPERTS,
        moderator_override_N_consecutive_answering_turn=MODERATOR_OVERRIDE_N_CONSECUTIVE_ANSWERING_TURN,
        node_expansion_trigger_count=NODE_EXPANSION_TRIGGER_COUNT,
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
    """Process citations in the article and add a references section.
    
    This function extracts citation information from the instance_dump,
    replaces citation markers in the article with clickable links,
    and adds a references section at the end of the article.
    
    Args:
        article (str): The original article text with citation markers.
        instance_dump (dict): The instance dump containing citation information.
            Expected to have a structure with knowledge_base.info_uuid_to_info_dict
            containing citation details.
            
    Returns:
        dict: A dictionary containing two versions of the processed article:
            - html (str): HTML version with clickable links and proper formatting.
            - markdown (str): Markdown version with proper citation links.
            
    Note:
        Citation markers in the article should be in the format [n] where n is a number.
        The function will attempt to match these markers with citation information
        from the instance_dump and replace them with clickable links.
    """
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
    """Debug function to print the structure of the instance_dump.json file.
    
    This function recursively traverses the instance_dump dictionary and prints
    its structure up to a specified maximum depth. It's useful for understanding
    the structure of complex nested dictionaries.
    
    Args:
        instance_dump (dict): The instance dump dictionary to debug.
        max_depth (int, optional): Maximum depth to traverse. Defaults to 2.
        current_depth (int, optional): Current depth in the recursion. Defaults to 0.
        path (str, optional): Current path in the dictionary. Defaults to "".
        
    Note:
        This is a helper function for development and debugging purposes.
        It prints the structure to the console and does not return any value.
    """
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
    """Run the Co-Storm algorithm in a separate thread.
    
    This function executes the Co-Storm algorithm for a given research topic,
    handling the conversation flow, user interactions, and final article generation.
    It manages the entire research session from initialization to completion.
    
    Args:
        session_id (str): The unique identifier for the client session.
        topic (str): The research topic to be explored by the Co-Storm algorithm.
        
    Note:
        This function is designed to be run in a separate thread to avoid blocking
        the main application thread. It communicates with the client through Socket.IO
        events to provide real-time updates and handle user interactions.
        
        The function creates an output directory for the session, initializes the
        Co-Storm runner with a custom callback handler, and manages the conversation
        flow until completion or the maximum number of turns is reached.
        
        Upon completion, it generates a final research article, processes citations,
        and saves all relevant data to the output directory.
    """
    try:
        # Create output directory for this session
        # Sanitize session_id to prevent path traversal attacks
        safe_session_id = re.sub(r'[^a-zA-Z0-9_-]', '', session_id)
        if safe_session_id != session_id:
            print(f"WARNING: Session ID contained unsafe characters and was sanitized: {session_id} -> {safe_session_id}")
            session_id = safe_session_id
            
        # Use os.path.join for proper path handling
        output_dir = os.path.join(".", "results", session_id)
        # Create directory with restricted permissions (0o700 = rwx------)
        os.makedirs(output_dir, exist_ok=True)
        # Ensure directory permissions are secure
        os.chmod(output_dir, 0o700)
        
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
    """Callback handler that updates the UI status during Co-Storm execution.
    
    This class extends the LocalConsolePrintCallBackHandler to provide real-time
    status updates to the web UI via Socket.IO events. It overrides various callback
    methods to send appropriate status messages at different stages of the Co-Storm
    algorithm execution.
    
    Attributes:
        socketio (SocketIO): The Socket.IO instance used to emit events to clients.
        session_id (str): The unique identifier for the client session.
    """
    
    def __init__(self, socketio, session_id):
        """Initialize the UIStatusCallbackHandler.
        
        Args:
            socketio (SocketIO): The Socket.IO instance used to emit events to clients.
            session_id (str): The unique identifier for the client session.
        """
        super().__init__()
        self.socketio = socketio
        self.session_id = session_id
    
    def _send_status_update(self, status):
        """Send status update to the UI.
        
        Args:
            status (str): The status message to be displayed in the UI.
        """
        self.socketio.emit('status_update', {'status': status}, room=self.session_id)
        print(f"Status update: {status}")
    
    def on_turn_policy_planning_start(self, **kwargs):
        """Run when the turn policy planning begins.
        
        This method is called at the start of planning the next conversation turn.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Planning next conversation turn...")
        super().on_turn_policy_planning_start(**kwargs)
    
    def on_expert_action_planning_start(self, **kwargs):
        """Run when the expert action planning begins.
        
        This method is called at the start of planning expert actions.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Planning expert actions...")
        super().on_expert_action_planning_start(**kwargs)
    
    def on_expert_action_planning_end(self, **kwargs):
        """Run when the expert action planning ends.
        
        This method is called when expert action planning is complete.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Expert actions planned")
        super().on_expert_action_planning_end(**kwargs)
    
    def on_expert_information_collection_start(self, **kwargs):
        """Run when the expert information collection starts.
        
        This method is called when experts begin collecting information.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Collecting information...")
        super().on_expert_information_collection_start(**kwargs)
    
    def on_expert_information_collection_end(self, info, **kwargs):
        """Run when the expert information collection ends.
        
        This method is called when experts have finished collecting information.
        
        Args:
            info (list): The collected information items.
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update(f"Collected {len(info)} pieces of information")
        super().on_expert_information_collection_end(info, **kwargs)
    
    def on_expert_utterance_generation_end(self, **kwargs):
        """Run when the expert utterance generation ends.
        
        This method is called when an expert has finished generating a response.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Generated expert response")
        super().on_expert_utterance_generation_end(**kwargs)
    
    def on_expert_utterance_polishing_start(self, **kwargs):
        """Run when the expert utterance polishing begins.
        
        This method is called when the system starts refining an expert's response.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Refining expert response...")
        super().on_expert_utterance_polishing_start(**kwargs)
    
    def on_mindmap_insert_start(self, **kwargs):
        """Run when the process of inserting new information into the mindmap starts.
        
        This method is called when the system begins organizing new information.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Organizing information...")
        super().on_mindmap_insert_start(**kwargs)
    
    def on_mindmap_insert_end(self, **kwargs):
        """Run when the process of inserting new information into the mindmap ends.
        
        This method is called when the system has finished organizing new information.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Information organized")
        super().on_mindmap_insert_end(**kwargs)
    
    def on_mindmap_reorg_start(self, **kwargs):
        """Run when the reorganization of the mindmap begins.
        
        This method is called when the system starts reorganizing the knowledge structure.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Reorganizing knowledge structure...")
        super().on_mindmap_reorg_start(**kwargs)
    
    def on_expert_list_update_start(self, **kwargs):
        """Run when the expert list update starts.
        
        This method is called when the system begins updating the list of experts.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Updating expert list...")
        super().on_expert_list_update_start(**kwargs)
    
    def on_article_generation_start(self, **kwargs):
        """Run when the article generation process begins.
        
        This method is called when the system starts generating the final research article.
        
        Args:
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update("Generating final article...")
        super().on_article_generation_start(**kwargs)
    
    def on_warmstart_update(self, message, **kwargs):
        """Run when the warm start process has an update.
        
        This method is called during the warm-up phase to provide status updates.
        
        Args:
            message (str): The warm-up status message.
            **kwargs: Additional keyword arguments passed by the Co-Storm engine.
        """
        self._send_status_update(f"Warming up: {message}")
        super().on_warmstart_update(message, **kwargs)

@app.route('/')
def index():
    """Render the main page of the application.
    
    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection to the Socket.IO server.
    
    This function is called when a new client connects to the Socket.IO server.
    It logs the connection and stores the session ID.
    """
    session_id = request.sid
    print(f"Client connected: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection from the Socket.IO server.
    
    This function is called when a client disconnects from the Socket.IO server.
    It logs the disconnection and cleans up any resources associated with the session.
    """
    session_id = request.sid
    print(f"Client disconnected: {session_id}")
    
    # Clean up session data if it exists
    if session_id in active_sessions:
        active_sessions[session_id]["completed"] = True
        del active_sessions[session_id]

@socketio.on('start_research')
def handle_start_research(data):
    """Start a new research session with the Co-Storm algorithm.
    
    This function initializes a new research session for the given topic,
    creates a separate thread to run the Co-Storm algorithm, and notifies
    the client that the research has started.
    
    Args:
        data (dict): A dictionary containing the research topic.
            Expected keys:
            - topic (str): The research topic to explore.
    """
    session_id = request.sid
    
    # Apply rate limiting
    if is_rate_limited(session_id):
        socketio.emit('error', {'message': 'Rate limit exceeded. Please try again later.'}, room=session_id)
        return
    
    # Validate input data
    if not isinstance(data, dict):
        socketio.emit('error', {'message': 'Invalid request format'}, room=session_id)
        return
        
    topic = data.get('topic', '')
    
    # Validate topic
    if not topic or not isinstance(topic, str):
        socketio.emit('error', {'message': 'Please provide a valid research topic'}, room=session_id)
        return
        
    # Limit topic length to prevent abuse
    max_topic_length = 500  # 500 characters should be more than enough
    if len(topic) > max_topic_length:
        socketio.emit('error', 
                     {'message': f'Topic too long (maximum {max_topic_length} characters)'}, 
                     room=session_id)
        return
    
    # Check if a session is already active for this client
    if session_id in active_sessions and not active_sessions[session_id].get("completed", True):
        socketio.emit('error', {'message': 'A research session is already in progress'}, room=session_id)
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
    """Handle a user message in an active research session.
    
    This function processes a message sent by the user during an active
    research session, stores it for processing by the Co-Storm algorithm,
    and echoes it back to the client.
    
    Args:
        data (dict): A dictionary containing the user message.
            Expected keys:
            - message (str): The message content from the user.
    """
    session_id = request.sid
    
    # Apply rate limiting
    if is_rate_limited(session_id):
        socketio.emit('error', {'message': 'Rate limit exceeded. Please try again later.'}, room=session_id)
        return
    
    # Validate input data
    if not isinstance(data, dict):
        socketio.emit('error', {'message': 'Invalid message format'}, room=session_id)
        return
        
    message = data.get('message', '')
    
    # Validate message content
    if not message or not isinstance(message, str):
        socketio.emit('error', {'message': 'Message cannot be empty'}, room=session_id)
        return
        
    # Limit message length to prevent abuse
    max_message_length = 5000  # 5000 characters should be more than enough
    if len(message) > max_message_length:
        socketio.emit('error', 
                     {'message': f'Message too long (maximum {max_message_length} characters)'}, 
                     room=session_id)
        return
    
    # Check for active session
    if session_id not in active_sessions:
        socketio.emit('error', {'message': 'No active research session'}, room=session_id)
        return
        
    if active_sessions[session_id]["completed"]:
        socketio.emit('error', {'message': 'Research session has ended'}, room=session_id)
        return
    
    # Store user message to be processed by the research thread
    active_sessions[session_id]["user_message"] = message
    active_sessions[session_id]["user_typing"] = False
    
    # Echo user message back to client
    socketio.emit('message', {'role': 'user', 'content': message}, room=session_id)

@socketio.on('typing_started')
def handle_typing_started():
    """Handle notification that the user has started typing.
    
    This function updates the session state to indicate that the user
    is currently typing, which can be used to delay AI responses.
    """
    session_id = request.sid
    
    if session_id in active_sessions and not active_sessions[session_id]["completed"]:
        active_sessions[session_id]["user_typing"] = True

@socketio.on('typing_stopped')
def handle_typing_stopped():
    """Handle notification that the user has stopped typing.
    
    This function updates the session state to indicate that the user
    has stopped typing, allowing the AI to continue with its responses.
    """
    session_id = request.sid
    
    if session_id in active_sessions and not active_sessions[session_id]["completed"]:
        active_sessions[session_id]["user_typing"] = False

if __name__ == '__main__':
    # Determine if we're in development or production mode
    debug_mode = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    
    # In production, don't use debug mode and only listen on localhost by default
    # unless explicitly configured to listen on all interfaces
    host = os.getenv('HOST', '127.0.0.1')  # Default to localhost
    port = int(os.getenv('PORT', 5001))
    
    # Log startup configuration
    print(f"Starting server in {'development' if debug_mode else 'production'} mode")
    print(f"Listening on {host}:{port}")
    
    # Run the application
    socketio.run(app, debug=debug_mode, host=host, port=port) 