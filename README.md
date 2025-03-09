# Co-Storm AI Research Web Application

This web application provides a user-friendly interface for the Co-Storm AI research algorithm. It allows users to input a research topic and engage in an interactive conversation with the AI to refine the research and generate a comprehensive article with citations.

## Features

- Interactive chat interface for collaborative research
- AI-guided research process that asks questions to refine the topic
- Intelligent waiting for human input with typing detection
- Final output as a well-structured article with citations
- Ability to download the research article as a Markdown file
- Real-time updates during the research process

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- A search engine API key (Serper, Bing, or Tavily)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd costorm-ui
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `secrets.toml` file based on the provided example:
   ```
   cp secrets.toml.example secrets.toml
   ```

4. Edit the `secrets.toml` file to add your API keys:
   ```
   OPENAI_API_KEY = "your-openai-api-key"
   OPENAI_API_TYPE = "openai"
   SERPER_API_KEY = "your-serper-api-key"  # Or use BING_SEARCH_API_KEY or TAVILY_API_KEY
   ```

## Usage

1. Start the web application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Enter a research topic in the input field and click "Start Research"

4. Engage with the AI in the chat interface to refine your research topic

5. Once the research is complete, the application will display the final article with citations

6. You can download the article as a Markdown file or start a new research session

## Environment Variables

You can configure the application using the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_API_TYPE`: API type (openai or azure)
- `RETRIEVER_TYPE`: Search engine to use (serper, bing, or tavily)
- `SERPER_API_KEY`: Serper API key (if using Serper)
- `BING_SEARCH_API_KEY`: Bing Search API key (if using Bing)
- `TAVILY_API_KEY`: Tavily API key (if using Tavily)
- `HUMAN_WAIT_TIME`: Number of seconds to wait for human input after each AI message (default: 10)

## Customization

You can modify the following constants in `app.py` to customize the application behavior:

- `NUM_TURNS`: Number of conversation turns in the research process
- `HUMAN_WAIT_TIME`: Number of seconds to wait for human input after each AI message

The application also features typing detection, which will pause the AI's responses as long as the user is typing, ensuring a more natural conversation flow.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This application uses the Co-Storm algorithm from the knowledge_storm package
- Built with Flask and Socket.IO for real-time communication


conda create -n storm python=3.11
conda activate storm
pip install -r requirements.txt


Had to fix bug in demo to reogranize

python run_costorm_gpt.py --output-dir ./results --enable_log_print --retriever serper


humanitarian response place name disambiguation

A blockchain (e.g., Ethereum or Hyperledger Fabric) can host an authoritative, consensus-based registry of geospatial identifiers, creating an immutable ledger of disambiguated place references. A decentralized ledger for geographic disambiguation, combined with off-chain spatial indexes.

Technical and Infrastructure Challenges, data volumes, Data Integration and Standardization