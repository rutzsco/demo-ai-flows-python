from dataclasses import dataclass

@dataclass
class FunctionCallResult:
    """A simple DTO for function call results."""
    name: str
    content: str