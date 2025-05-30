from dataclasses import dataclass, field
from typing import List

@dataclass
class Source:
    quote: str
    title: str
    url: str

@dataclass
class ExecutionStep:
    name: str
    content: str
    start_time: str | None = None
    end_time: str | None = None

@dataclass
class ExecutionDiagnostics:
    steps: List[ExecutionStep] = field(default_factory=list)

@dataclass
class RequestResult:
    """A simple DTO for function call results."""
    content: str
    execution_diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    intermediate_steps: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    files: list[Source] = field(default_factory=list)
    thread_id: str = None
    code_content: str = None # Add new field for code content

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatRequest:
    messages: List[ChatMessage] = field(default_factory=list)

@dataclass
class ChatThreadRequest:
    message: str
    file: str = None
    thread_id: str = None

@dataclass
class Source:
    title: str
    quote: str = None
    url: str = None

@dataclass
class FileReference:
    id: str
    url: str = None

@dataclass
class AgentCreateRequest:
    instructions: str
    name: str
    model: str