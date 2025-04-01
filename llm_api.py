import litellm
from typing import Dict, List, Optional, Union
from litellm.integrations.custom_logger import CustomLogger
from llm_logger import logger
import os
import io
import base64
import numpy as np
import cv2
from PIL import Image

class CustomHandler(CustomLogger):
    def __init__(self):
        self.user_id = None
        super().__init__()

    def log_pre_api_call(self, model, messages, kwargs):
        # log uid, model, messages, kwargs
        if self.user_id:
            logger.info(f"User ID: {self.user_id} Model: {model} Input Messages: {messages}")
        else:
            logger.info(f"Model: {model} Input Messages: {messages}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        if self.user_id:
            logger.info(f"User ID: {self.user_id}, Total Tokens: {response_obj.usage.total_tokens}")
        else:
            logger.info(f"Total Tokens: {response_obj.usage.total_tokens}")
        if self.user_id:
            logger.info(f"User ID: {self.user_id}, Output Messages: {response_obj.choices[0].message.content}")
        else:
            logger.info(f"Output Messages: {response_obj.choices[0].message.content}")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        logger.error(f"Error: {response_obj}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if self.user_id:
            logger.info(f"User ID: {self.user_id}, Total Tokens: {response_obj.usage.total_tokens}")
        else:
            logger.info(f"Total Tokens: {response_obj.usage.total_tokens}")
        if self.user_id:
            logger.info(f"User ID: {self.user_id}, Output Messages: {response_obj.choices[0].message.content}")
        else:
            logger.info(f"Output Messages: {response_obj.choices[0].message.content}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        logger.error(f"Error: {response_obj}")



class RunningInstruction:
    def __init__(self):
        self.instructions = {
            "openai": [
                "Step 1: prepare your OPENAI_API_KEY first and set the environment variable in function 'prepare_env_for_agent_type'"
            ]

        }
        self.url_set = {
            "openai": "https://api.zhizengzeng.com/v1/chat/completions"
        }
        self.type = None
        self.base_url = None

    def prepare_env_for_agent_type(self, agent_type):
        if "openai" in agent_type:
            self.type = "openai"
        else:
            print(f"{agent_type} is not supported. Please re-check.")
            exit()

        self.base_url = self.url_set[self.type]

    def show_instructions(self):
        assert self.type is not None, "You need to invoke prepare_env_for_agent_type first."
        guide = self.instructions[self.type]
        print("Pre-requisites for running this program")
        for i in guide:
            print(i)
        print("-------------------------------------------------------------")


def read_prompt_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip() 
    except FileNotFoundError:
        print(f"Error: file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None



class LLM_API(RunningInstruction):
    # Support multi-modal language models, such as gpt-4o-mini and gpt-4o.
    def __init__(self, agent_type: str, model_name: str, temperature: float = 0.5):
        super().__init__()
        self.prepare_env_for_agent_type(agent_type)
        self.show_instructions()

        self.model_name = model_name
        self.logger = CustomHandler()
        
        self.temperature = temperature

    def generate(self, prompt: Optional[Union[str, list[str]]]):
        if isinstance(prompt, list):
            prompt = prompt[0]
        messages = [{"role": "user", "content": prompt}]
        litellm.callbacks = [self.logger]
        
        response = litellm.completion(model=self.model_name, messages=messages, num_retries=3, base_url=self.base_url, temperature=self.temperature)
        return response.choices[0].message.content

    def completion(self, messages: List[Dict[str, str]], user_id: str = None):
        if user_id:
            self.logger.user_id = user_id
        litellm.callbacks = [self.logger]
        response = litellm.completion(model=self.model_name, messages=messages, num_retries=3, base_url=self.base_url, temperature=self.temperature)
        return response.choices[0].message.content
    
    async def acompletion(self, messages: List[Dict[str, str]], user_id: str = None):
        if user_id:
            self.logger.user_id = user_id
        litellm.callbacks = [self.logger]
        response = await litellm.acompletion(model=self.model_name, messages=messages, num_retries=3, base_url=self.base_url, temperature=self.temperature)
        return response.choices[0].message.content

    def json_completion(self, messages: List[Dict[str, str]], user_id: str = None):
        if user_id:
            self.logger.user_id = user_id
        litellm.callbacks = [self.logger]
        response = litellm.completion(model=self.model_name, response_format={ "type": "json_object" }, messages=messages, num_retries=3, base_url=self.base_url, temperature=self.temperature)
        return response.choices[0].message.content
    
    async def ajson_completion(self, messages: List[Dict[str, str]], user_id: str = None):
        if user_id:
            self.logger.user_id = user_id
        litellm.callbacks = [self.logger]
        response = await litellm.acompletion(model=self.model_name, response_format={ "type": "json_object" }, messages=messages, num_retries=3, base_url=self.base_url, temperature=self.temperature)
        return response.choices[0].message.content
    
    async def astream_completion(self, messages: List[Dict[str, str]], user_id: str = None):
        if user_id:
            self.logger.user_id = user_id
        litellm.callbacks = [self.logger]
        response = await litellm.acompletion(model=self.model_name, messages=messages, stream=True, stream_options={"include_usage": True}, num_retries=3, base_url=self.base_url, temperature=self.temperature)
        async for chunk in response: 
            if chunk['choices'][0]['delta']['content'] is None:
                continue
            yield chunk['choices'][0]['delta']['content']




def encode_image(image):

    # image path
    if isinstance(image, str): 
        with open(image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # cv2
    elif isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.jpg', image) 
        base64_image = base64.b64encode(buffer).decode('utf-8')  
    
    # PIL
    elif isinstance(image, Image.Image):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

    else:
        raise TypeError

    return base64_image
    