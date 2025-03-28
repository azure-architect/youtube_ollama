import json
import aiohttp
from typing import Dict, Any, Optional, List, TypeVar, Generic, Type
from pydantic import BaseModel

# Generic type for the result model
T = TypeVar('T', bound=BaseModel)

class BaseAgent(Generic[T]):
    """Base agent class that handles common functionality for Ollama API interactions."""
    
    def __init__(self, 
                 model_name: str = "mistral:latest",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 num_ctx: int = 4096,
                 result_model: Type[T] = None):
        """
        Initialize the base agent with common parameters.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
            temperature: Temperature setting for generation
            num_ctx: Context window size
            result_model: Pydantic model class for the expected result
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.result_model = result_model
        
        # Define the schema based on the result model if provided
        self.schema = self._get_schema_from_model() if result_model else None
    
    def _get_schema_from_model(self) -> Dict[str, Any]:
        """Generate JSON schema from the Pydantic model."""
        if not self.result_model:
            return {}
        return self.result_model.model_json_schema()
    
    async def call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call the Ollama API with the given prompt and return the raw response."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx
                }
            }
            
            # Add format parameter if schema is defined
            if self.schema:
                payload["format"] = self.schema
            
            try:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error ({response.status}): {error_text}")
                    
                    return await response.json()
            except Exception as e:
                print(f"Error calling Ollama API: {e}")
                raise
    
    async def parse_structured_output(self, result: Dict[str, Any]) -> T:
        """Parse the structured output from the Ollama response."""
        try:
            # Parse the structured output from the response
            structured_data = json.loads(result["response"])
            
            # Convert to the expected result type
            if self.result_model:
                return self.result_model(**structured_data)
            return structured_data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing structured output: {e}")
            print(f"Raw response: {result['response']}")
            
            # Implement fallback parsing
            return await self.fallback_parsing(result["response"])
    
    async def fallback_parsing(self, raw_text: str) -> T:
        """
        Fallback parsing method to extract structured data from raw text.
        Subclasses should override this method to implement specific fallback parsing.
        """
        raise NotImplementedError("Fallback parsing must be implemented by subclasses")
    
    async def _prepare_prompt(self, input_data: Any) -> str:
        """
        Prepare the prompt for the agent.
        Subclasses should override this method to implement specific prompt preparation.
        """
        raise NotImplementedError("Prompt preparation must be implemented by subclasses")
    
    async def run(self, input_data: Any, message_history: Optional[List[Dict[str, str]]] = None) -> T:
        """Run the agent with the given input data."""
        try:
            # Prepare the prompt
            prompt = await self._prepare_prompt(input_data)
            
            # Call the Ollama API
            result = await self.call_ollama(prompt)
            
            # Parse the result
            return await self.parse_structured_output(result)
        except Exception as e:
            print(f"Error running agent: {e}")
            # Return a default instance of the result model if possible
            if self.result_model:
                # This assumes the model has default values or optional fields
                # You might need to provide minimum required fields depending on your model
                return self.result_model()
            raise