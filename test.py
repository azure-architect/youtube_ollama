import asyncio
import httpx
import json
import logging
import sys

# Set up logging to match your application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_ollama_as_agent():
    """Test Ollama API access exactly as the agent would access it"""
    
    # Use the same model and URL as your agent
    model_name = "mistral:latest"  # Same default as your agent
    base_url = "http://localhost:11434"
    temperature = 0.1
    num_ctx = 4096
    
    # Short sample prompt
    prompt = """
    You are an expert analyzer of video content. Extract key insights from this transcript.
    
    TRANSCRIPT:
    This is a short test transcript about simple websites that make money.
    
    Please provide a JSON response with:
    1. A summary
    2. Key points
    
    Format as valid JSON.
    """
    
    try:
        # Use the exact same code as your agent's _get_model_response method
        url = f"{base_url}/api/generate"
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "num_ctx": num_ctx
        }
        
        logger.info(f"Sending request to {url}")
        logger.info(f"Using model: {model_name}")
        logger.info(f"Payload: {json.dumps(payload)[:200]}...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=60.0)
            logger.info(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"SUCCESS! Response: {json.dumps(result)[:200]}...")
                return result.get("response", "")
            else:
                logger.error(f"Failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return f"Error: {response.status_code}"
                
    except Exception as e:
        logger.error(f"Error getting model response: {type(e).__name__}: {e}")
        return f"Error: {str(e)}"
    
    # Also check available models
    try:
        logger.info("\nVerifying available models...")
        models_url = f"{base_url}/api/tags"
        
        async with httpx.AsyncClient() as client:
            models_response = await client.get(models_url)
            
            if models_response.status_code == 200:
                models = models_response.json()
                model_names = [m['name'] for m in models.get('models', [])]
                logger.info(f"Available models: {model_names}")
                
                if model_name in model_names:
                    logger.info(f"✓ Model '{model_name}' is available")
                else:
                    logger.warning(f"✗ Model '{model_name}' is NOT in the available models list")
                    logger.info("Available models are: " + ", ".join(model_names))
            else:
                logger.error(f"Failed to list models: {models_response.status_code}")
    except Exception as e:
        logger.error(f"Error listing models: {type(e).__name__}: {e}")

if __name__ == "__main__":
    result = asyncio.run(test_ollama_as_agent())
    print("\nFinal result from test:")
    print(result)