from typing import Optional
from dataclasses import dataclass


@dataclass
class Template:

    name: Optional[str] = None
    prompt: Optional[str] = None

    def __post_init__(self):

        if self.prompt is not None:
            return

        if self.name == "default":
            self.prompt = "{query}"

        elif self.name == "alpaca":
            self.prompt = "### Instruction:\n{query}\n\n### Response:\n"

        elif self.name == "vicuna":
            self.prompt = "USER: {query} ASSISTANT: "

        elif self.name == "ziya":
            self.prompt = "<human>:{query}\n<bot>:"

        else:
            raise NotImplementedError

    def get_prompt(self, query: str) -> str:
        return self.prompt.format(query=query)

    @classmethod
    def from_str(cls, tmp: str):
        return cls(prompt=tmp)
