"""
Test script for the Serper API integration.

This script demonstrates how to make a basic search query to the Google Serper API.
It sends a simple search query and prints the raw JSON response.

Note:
    This is a standalone test script and requires a valid Serper API key.
"""

import http.client
import json

# API endpoint configuration
SERPER_HOST = "google.serper.dev"
SERPER_API_KEY = "b2af04d2149e997c5b923932db9a5c7e96bfb9ef"

def test_serper_search(query="apple inc"):
    """
    Perform a test search using the Serper API.
    
    Args:
        query (str, optional): The search query to send to Serper. Defaults to "apple inc".
        
    Returns:
        str: The decoded JSON response from the Serper API.
    """
    # Establish connection to the Serper API
    conn = http.client.HTTPSConnection(SERPER_HOST)
    
    # Prepare the request payload
    payload = json.dumps({"q": query})
    
    # Set up request headers with API key
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Send the request
    conn.request("POST", "/search", payload, headers)
    
    # Get the response
    res = conn.getresponse()
    data = res.read()
    
    # Return the decoded response
    return data.decode("utf-8")

if __name__ == "__main__":
    # Execute the test search and print the results
    result = test_serper_search()
    print(result)