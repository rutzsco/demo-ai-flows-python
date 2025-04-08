from dataclasses import dataclass, field
from typing import List

@dataclass
class ExecutionStep:
    name: str
    content: str

@dataclass
class ExecutionDiagnostics:
    steps: List[ExecutionStep] = field(default_factory=list)

@dataclass
class RequestResult:
    """A simple DTO for function call results."""
    content: str
    execution_diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    intermediate_steps: list[str] = field(default_factory=list)  

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatRequest:
    messages: List[ChatMessage] = field(default_factory=list)
