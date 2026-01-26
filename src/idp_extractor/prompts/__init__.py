from .base import PromptTemplate

# Import specific prompt versions
from .passeport.v1 import PassportPromptV1

# Default version
DEFAULT_PROMPT = PassportPromptV1()