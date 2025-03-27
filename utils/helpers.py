import subprocess
import requests
import time
import json

def ensure_ollama_model(model_name="llama3-groq-tool-use:latest"):
    """
    Ensures that Ollama is running and the specified model is available.
    Returns True if successful, False otherwise.
    """
    # Check if Ollama server is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("Ollama server is not responding correctly. Starting Ollama...")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait for server to start
            time.sleep(5)
    except requests.exceptions.ConnectionError:
        print("Ollama server is not running. Starting Ollama...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for server to start
        time.sleep(5)
    
    # Check if the model is available
    try:
        response = requests.get("http://localhost:11434/api/tags")
        available_models = [tag["name"] for tag in response.json()["models"]]
        
        if model_name not in available_models:
            print(f"Model {model_name} not found. Pulling the model...")
            result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
            if "error" in result.stderr.lower():
                print(f"Error pulling model: {result.stderr}")
                return False
            print(f"Model {model_name} pulled successfully.")
        else:
            print(f"Model {model_name} is available.")
        
        return True
    except Exception as e:
        print(f"Error checking or pulling Ollama model: {e}")
        return False