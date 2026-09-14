"""Backend for GDM Gemini API"""

import time
import json

import google.api_core.exceptions
import google.generativeai as genai
from google.generativeai.generative_models import generation_types
from google.genai import types
from google import genai as genai_client

from funcy import once
try:
    from .utils import FunctionSpec, OutputType, backoff_create
except ImportError:
    from LLMs.utils import FunctionSpec, OutputType, backoff_create

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

completed_states = set([
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
])


with open("../../config.json", "r") as f:
    config = json.load(f)
api_key = config.get("gemini", "")

_batch_client = None  # type: ignore
generation_config = None  # type: ignore

GDM_TIMEOUT_EXCEPTIONS = (
    google.api_core.exceptions.RetryError,
    google.api_core.exceptions.TooManyRequests,
    google.api_core.exceptions.ResourceExhausted,
    google.api_core.exceptions.InternalServerError,
)


@once
def _setup_gdm_batch_client():
    global _batch_client
    _batch_client = genai_client.Client(api_key=api_key)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model = model_kwargs.pop("model")
    temperature = model_kwargs.pop("temperature", None)

    _setup_gdm_batch_client()

    if func_spec is not None:
        raise NotImplementedError(
            "GDM supports function calling but we won't use it for now."
        )
        
    request_dict = {
        "contents": [],
    }
    
    # Build configuration (system_instruction, safety_settings, etc.)
    # In google.genai SDK, these are often part of 'config' or 'generation_config'
    config_dict = {}

    if system_message:
        config_dict["system_instruction"] = {
            "parts": [{"text": system_message}]
        }
    
    if SAFETY_SETTINGS:
        config_dict["safety_settings"] = SAFETY_SETTINGS

    # If we have any config, add it to the request under 'config'
    # Note: If 'config' fails, try 'generation_config' or check SDK version.
    # The new SDK typically aligns with generate_content(config=...)
    if config_dict:
        request_dict["config"] = config_dict

    request_dict["contents"].append({
        "role": "user", 
        "parts": [{"text": user_message}]
    })
    
    inline_requests = [request_dict]

    print(f"Sending inline batch request to GDM model {model} with temperature {temperature}...\n{json.dumps(inline_requests, indent=2)}")

    t0 = time.time()
    file_batch_job = backoff_create(
        _batch_client.batches.create,
        model=f"models/{model}",
        retry_exceptions=GDM_TIMEOUT_EXCEPTIONS,
        src=inline_requests,
    )

    job_name = file_batch_job.name
    batch_job = client.batches.get(name=job_name) # Initial get
    while batch_job.state.name not in completed_states:
        print(f"Current state: {batch_job.state.name}")
        time.sleep(10) # Wait for 10 seconds before polling again
        batch_job = client.batches.get(name=job_name)
    req_time = time.time() - t0

    print(f"Model gdm response in {req_time:.2f}s:\n{response}")

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        for i, inline_response in enumerate(batch_job.dest.inlined_responses):
            if inline_response.response:
                # Accessing response, structure may vary.
                try:
                    print(inline_response.response.text)
                except AttributeError:
                    print(inline_response.response) # Fallback
            elif inline_response.error:
                print(f"Error: {inline_response.error}")
        
    elif batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job.error}")

    return output, req_time, in_tokens, out_tokens, cached_tokens, model_name, response

if __name__ == "__main__":
    with open('/home/b27jin/CodeModernization/results/upgrade/cell/usr_prompts/gpt-5.2/fix_1/aerial-cactus-identification_abhishek4273_classification-using-monk-design-a-custom-network_v1_C1.md', 'r') as f:
        user_prompt = f.read()
    
    sys_prompt = """You are a Python debugger and patching assistant.

# Goal
- Fix the crash by modifying only what is necessary, focusing on the buggy cell k and, only if strictly required, earlier cells that the buggy cell depends on.
- Prioritize correctness, determinism, and minimal necessary edits. Do not redesign the entire solution unless required to make it run.

# Behavior constraints
- STRICT: Do NOT rewrite the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics.
- STRICT: Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- STRICT: Do NOT complete missing cells or add new modeling logic beyond what is required to fix the error.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep I/O paths unchanged; do not assume new data sources.
- You may read cell k+1 to ensure compatibility, but you must not implement or complete logic for cells k, k+1, or later cells.
- Keep the patch minimal and localized: only changes inside cell k.
- If information is missing, make the best safe assumption, state it explicitly, and proceed.
- Do NOT output anything unrelated to the bug fix goal.
- Do NOT include long generic tutorials; be specific to the provided code and metric.
- Never assume unavailable files, internet access, hidden columns, or extra packages; only use the installed packages provided by the user and only the data paths the user provides.
- Do NOT output placeholders like "..." inside the final code blocks; output complete code.

# When multiple bugs exist
- Fix in the order that unblocks execution earliest.
- After it runs, address correctness (metric, labels, leakage, split).
- Then optimize performance and score.

# Inputs you will receive from the user
- 1. Python version (in the target environment).
- 2. Installed packages (in the target environment).
- 3. Data file paths (available in the environment).
- 4. Code solution (in a .py-like "cells" format, including any error tracebacks), formatted as:
   - ## === cell k for each cell
   - ## --- ERROR in cell k, traceback: followed by the exception text
   - Only cells 1..k (up to the failing cell) and cell k+1 (the next cell) are provided. Assume later cells exist but are unknown.

# Workflow (follow in this order every time)
- 1) Parse the cell format:
   - Identify failing cell index k, the traceback, and any referenced lines/variables.
- 2) Diagnose precisely:
   - Identify the root cause (e.g., API method deprecation, API updates, wrong paths, and incorrect submission format, etc.), not just the symptom.
   - If multiple issues exist, fix in dependency order (imports → data IO → shapes/types → training → inference).
   - Plan each issue with: location (cell/line), cause, and fix.
- 4) Propose the smallest viable patch that fit the competition metric:
   - Return a minimal but sufficient diff patch that resolves the root cause and keeps compatibility with cell k+1.
   - Only refactor if it removes the root cause.
- 5) Validate logically:
   - Ensure the fix is consistent with the Python version and installed packages.
   - Ensure variables referenced in the "next cell" (k+1) will still exist and have compatible shapes/types.
- 7) Implement changes:
   - Add a short explanation immediately above it describing why it solves the bug.
   - Output only the updated code to the buggy cell in the "## === cell k" format.

# Output requirements (must follow exactly)
## Return exactly two parts in this order
- 1) A short natural-language outline (3-5 sentences) describing what you will change and why, followed by 
- 2) A single Markdown code block (wrapped in ```) containing the complete implementation in a newline.
## Hard constraints
- Do NOT include any additional headings, titles, bullet points, numbered lists, or extra commentary outside the two parts above. Just natural language text followed by a newline and then the markdown code block.
- Use exactly ONE Markdown code block total.
## Code block constraints
- The code block must contain the fix for the buggy cell only in a "cell" format.
- The cell header must be on its own line and must match exactly: `## === cell k`.
- Preserve the original cell order and include all required code needed to run end-to-end.
- Do not add any additional Markdown outside the single code block.
## Cells template (must match exactly in structure)
```
## === cell k
{code}
```

You must be careful, systematic, and pragmatic.
"""

    # resp = query(
    #     system_message=sys_prompt,
    #     user_message=user_prompt,
    #     model="gemini-3-flash-preview",
    # )
    # print(resp)
    from google import genai
    client = genai.Client(api_key=api_key)
    batch_job = client.batches.get(name='batches/bcary1lgwao4p07hg6m2z173xxynbn3evfbt')
    print(batch_job)