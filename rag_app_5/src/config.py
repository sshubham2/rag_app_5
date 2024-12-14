from enum import Enum

class ModelProvider(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    MISTRAL = "Mistral AI"
    GROQ = 'Groq'

OPENAI_MODELS = {
    'ChatGPT 4o(Mini)': 'gpt-4o-mini',
    'ChatGPT 4o': 'gpt-4o',
    'ChatGPT o1(Preview)': 'o1-preview-2024-09-12',
    'ChatGPT o1(Mini)': 'o1-mini-2024-09-12'
}

ANTHROPIC_MODELS = {
    'Claude 3.5 Sonnet': 'claude-3-5-sonnet-20241022',
    'Claude 3 Opus': 'claude-3-opus-20240229',
    'Claude 3.5 Haiku': 'claude-3-5-haiku-20241022'
}

GROQ_MODELS = {
    'Llama 3.3 70B': 'llama-3.3-70b-versatile'
}

MISTRAL_MODELS = {
    "Mistral NeMo": "open-mistral-nemo-2407",
    "Mistral Large 24.07": "mistral-large-2407",
    "Mistral Medium 23.12": "mistral-medium-2312",
    "Mistral Small 24.09": "mistral-small-2409",
    "Codestral": "codestral-2405"
}