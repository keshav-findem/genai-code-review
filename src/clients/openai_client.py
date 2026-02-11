"""
Este módulo contém a classe OpenAIClient, que é usada para interagir com a API do OpenAI.
A classe OpenAIClient pode ser usada para gerar respostas de um modelo especificado do OpenAI.
"""

import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenAIClient:
    """
    A client for interacting with the OpenAI API to generate responses using a specified model.
    """

    def __init__(self, model, temperature, max_completion_tokens):
        """
        Initialize the OpenAIClient with API key, model, temperature, and max tokens.

        Args:
            api_key (str): The OpenAI API key.
            model (str): The OpenAI model to use.
            temperature (float): The sampling temperature.
            max_completion_tokens (int): The maximum number of tokens to generate.
        """
        try:
            self.client = OpenAI()
            self.model = model
            self.temperature = temperature
            self.max_completion_tokens = max_completion_tokens
            logging.info(
                "OpenAI client initialized successfully, "
                "Model: %s, temperature: %s, max tokens: %s",
                self.model,
                self.temperature,
                self.max_completion_tokens
            )
        except Exception as e:
            logging.error("Error initializing OpenAI client: %s", e)
            raise

    def generate_response(self, prompt):
        """
        Generate a response from the OpenAI model based on the given prompt.

        Args:
            prompt (str): The prompt to send to the OpenAI API.

        Returns:
            str: The generated response from the OpenAI model.

        Raises:
            Exception: If there is an error generating the response.
        """
        try:
            logging.info("Generating response from OpenAI model.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Developer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )
            logging.info("Response generated successfully.")
            
            # Debug logging for response structure
            if response.choices:
                choice = response.choices[0]
                logging.info("Response choice finish_reason: %s", choice.finish_reason)
                if hasattr(choice.message, 'content'):
                    content = choice.message.content
                    logging.info("Message content type: %s", type(content))
                    logging.info("Message content is None: %s", content is None)
                    if content:
                        logging.info("Response content length: %d characters", len(content))
                    else:
                        logging.warning("OpenAI message.content is None or empty")
                        # Check for refusal or other fields
                        if hasattr(choice.message, 'refusal') and choice.message.refusal:
                            logging.error("OpenAI refused the request: %s", choice.message.refusal)
                            return f"OpenAI refused the request: {choice.message.refusal}"
                        logging.warning("Full message object: %s", choice.message)
                        return "No review generated. The model returned an empty response."
                    return content
                else:
                    logging.error("Response message has no content attribute")
                    return "Error: Unexpected response format from OpenAI"
            else:
                logging.error("Response has no choices")
                return "Error: No response choices returned from OpenAI"
        except Exception as e:
            logging.error("Error generating response from OpenAI model: %s", e)
            raise
