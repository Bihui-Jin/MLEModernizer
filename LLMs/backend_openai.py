"""Backend for OpenAI API."""

import json
import time

from LLMs.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai

_client: openai.OpenAI = None  # type: ignore


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

with open("../config.json", "r") as f:
    config = json.load(f)
api_key = config.get("openai", "")

@once
def _setup_openai_client(model: str):
    global _client
    if model == "qwen3-max":
        _client = openai.OpenAI(base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1', max_retries=5)
    elif "gpt-oss" in model or "claude" in model:
        _client = openai.OpenAI(base_url='https://openrouter.ai/api/v1', api_key=config.get("openrouter", ""), max_retries=5)
        # _client = openai.OpenAI(base_url='http://localhost:8000/v1', api_key=api_key, max_retries=5)
    else:
        _client = openai.OpenAI(api_key=api_key, max_retries=5)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model = "openai/" + model_kwargs.get("model", "") if "gpt" in model_kwargs.get("model", "") else model_kwargs.get("model", "")
    model = "anthropic/" + model_kwargs.get("model", "") if "claude" in model_kwargs.get("model", "") else model_kwargs.get("model", "")
    model_kwargs["model"] = model
    _setup_openai_client(model)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)
    
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    # filtered_kwargs["provider"] = {
    #     'only': 'google-vertex/global',
    # }

    # OpenRouter provider routing (not a standard OpenAI SDK parameter)
    raw_model = model_kwargs.get("model", "")
    if "gpt-oss" in raw_model or "claude" in raw_model:
        extra_body = dict(filtered_kwargs.pop("extra_body", None) or {})
        provider_prefs = filtered_kwargs.pop("provider", None) or {
            "only": ["google-vertex/global"],
        } if "gpt-oss" in raw_model else {
            "only": ["anthropic"],
        }
        extra_body["provider"] = provider_prefs
        filtered_kwargs["extra_body"] = extra_body
    if "gpt-5" in raw_model:
    #     extra_body = dict(filtered_kwargs.pop("extra_body", None) or {})
    #     extra_body["reasoning"] = {"enabled": True}
    #     filtered_kwargs["extra_body"] = extra_body
        filtered_kwargs["service_tier"] = "flex"

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    res_id = completion.id
    choice = completion.choices[0]
    # print(f"Model openai response in {req_time:.2f}s:\n{completion}")

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:

            function_args = choice.message.tool_calls[0].function.arguments
            print(f"Raw function arguments: {repr(function_args)}")
            
            # Clean up the function arguments string
            function_args_cleaned = function_args.strip()

            # Fix common JSON formatting issues
            # 1. Handle incomplete values like "metric": , -> "metric": null,
            import re
            function_args_cleaned = re.sub(r':\s*,', ': null,', function_args_cleaned)
            function_args_cleaned = re.sub(r':\s*}', ': null}', function_args_cleaned)

            # 2. Ensure proper JSON boolean and null values (keep as lowercase for valid JSON)
            # Don't convert to Python style here - JSON parser expects lowercase
            function_args_cleaned = function_args_cleaned.replace(': True', ': true')
            function_args_cleaned = function_args_cleaned.replace(': False', ': false')
            function_args_cleaned = function_args_cleaned.replace(':True', ':true')
            function_args_cleaned = function_args_cleaned.replace(':False', ':false')

            # Also handle None -> null
            function_args_cleaned = function_args_cleaned.replace(': None', ': null')
            function_args_cleaned = function_args_cleaned.replace(':None', ':null')
            
            # Try to parse the JSON
            output = json.loads(function_args_cleaned)
        except json.JSONDecodeError as e:
            print(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            print(f"Error position: line {e.lineno}, column {e.colno}")
            
            # Try to extract partial JSON or provide a fallback
            function_args = choice.message.tool_calls[0].function.arguments

            try:
                # Remove any potential control characters or invisible characters
                import re
                cleaned_args = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', function_args)
                
                # Fix incomplete JSON values
                cleaned_args = re.sub(r':\s*,', ': null,', cleaned_args)
                cleaned_args = re.sub(r':\s*}', ': null}', cleaned_args)
                
                # Normalize Python-style literals to JSON-style
                cleaned_args = re.sub(r'\bTrue\b', 'true', cleaned_args)
                cleaned_args = re.sub(r'\bFalse\b', 'false', cleaned_args)
                cleaned_args = re.sub(r'\bNone\b', 'null', cleaned_args)

                # Find the first { and last }
                start = cleaned_args.find('{')
                end = cleaned_args.rfind('}')
                
                if start != -1 and end != -1 and end > start:
                    json_str = cleaned_args[start:end+1]
                    print(f"Attempting to parse cleaned JSON: {json_str}")
                    output = json.loads(json_str)
                else:
                    raise e
            except json.JSONDecodeError:
                print(f"Failed to extract valid JSON in: {function_args}")
                # Provide a fallback response that matches the expected schema
                output = {
                    "is_bug": True,
                    "has_csv_submission": False,
                    "summary": f"JSON parsing error in function response: {str(e)}",
                    "metric": None,
                    "lower_is_better": True
                }

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens
    model_name = completion.model
    # prompt_tokens_details may be:
    # - None
    # - a dict
    # - a PromptTokensDetails object (not subscriptable)
    ptd = getattr(completion.usage, "prompt_tokens_details", None)
    if isinstance(ptd, dict):
        cached_tokens = ptd.get("cached_tokens", 0) or 0
    else:
        cached_tokens = getattr(ptd, "cached_tokens", 0) if ptd is not None else 0

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, cached_tokens, model_name, completion
