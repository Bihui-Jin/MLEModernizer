"""Backend for GDM Gemini API"""

import time
import json

import google.api_core.exceptions
import google.generativeai as genai
from google.generativeai.generative_models import generation_types
from google.genai import types

from funcy import once
from .utils import FunctionSpec, OutputType, backoff_create


with open("../config.json", "r") as f:
    config = json.load(f)
api_key = config.get("gemini", "")

gdm_model = None  # type: ignore
generation_config = None  # type: ignore

GDM_TIMEOUT_EXCEPTIONS = (
    google.api_core.exceptions.RetryError,
    google.api_core.exceptions.TooManyRequests,
    google.api_core.exceptions.ResourceExhausted,
    google.api_core.exceptions.InternalServerError,
)
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


@once
def _setup_gdm_client(model_name: str, system_instruction: str | None = None):
    global gdm_model
    # global generation_config

    genai.configure(api_key=api_key)
    gdm_model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
    # generation_config = types.GenerateContentConfig()


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model = model_kwargs.pop("model")
    temperature = model_kwargs.pop("temperature", None)

    _setup_gdm_client(model, system_instruction=system_message)

    if func_spec is not None:
        raise NotImplementedError(
            "GDM supports function calling but we won't use it for now."
        )

    t0 = time.time()
    response: generation_types.GenerateContentResponse = backoff_create(
        gdm_model.generate_content,
        retry_exceptions=GDM_TIMEOUT_EXCEPTIONS,
        contents=user_message,
        # config=generation_config,
        safety_settings=SAFETY_SETTINGS,
    )
    req_time = time.time() - t0

    # print(f"Model gdm response in {req_time:.2f}s:\n{response}")

    res_id = getattr(response, "response_id", None) 
    if not res_id and hasattr(response, "result"):
        res_id = getattr(response.result, "response_id", None)
    # print(f"Gemini Response ID: {res_id}")

    if response.prompt_feedback.block_reason:
        output = str(response.prompt_feedback)
    else:
        output = response.text

    usage = getattr(response, "usage_metadata", None)
    in_tokens = getattr(usage, "prompt_token_count", 0) or 0
    out_tokens = getattr(usage, "candidates_token_count", 0) or 0
    cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

    # if hasattr(response, 'usageMetadata'):
    #     usage = response.usageMetadata
    #     in_tokens = getattr(usage, 'promptTokenCount', 0) or 0
    #     out_tokens = getattr(usage, 'candidatesTokenCount', 0) or 0
    #     cached_tokens = getattr(usage, 'cachedContentTokenCount', 0) or 0

    model_name = getattr(response, "model_version", None) or model

    return output, req_time, in_tokens, out_tokens, cached_tokens, model_name, response
