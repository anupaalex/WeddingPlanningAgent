# gemini_adapter.py
from google import genai
from google.genai import types

from wedding_rag.settings import load_secrets
def make_gemini():
    client = genai.Client(api_key=load_secrets().gemini_api_key.get_secret_value(),
                          http_options=types.HttpOptions(timeout=30_000))   # milliseconds: check your SDK version's docs

    def count(model: str, text: str) -> int:
        return client.models.count_tokens(model=model, contents=text).total_tokens

    def call(model, text, max_out, system=None, temperature=0.0, response_schema=None, tools=None):
            kwargs = dict(max_output_tokens=max_out, temperature=temperature, system_instruction=system,
                        response_mime_type="application/json" if response_schema else None,
                        response_schema=response_schema)
            if tools:
                kwargs["tools"] = [types.Tool(function_declarations=tools)]
                kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
            return client.models.generate_content(model=model, contents=text,
                                                config=types.GenerateContentConfig(**kwargs))
    return count, call

    