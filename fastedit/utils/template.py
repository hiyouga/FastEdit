from typing import Optional
from dataclasses import dataclass


@dataclass
class Template:

    name: Optional[str] = None
    prompt: Optional[str] = None

    def __post_init__(self):

        if self.prompt is not None:
            assert "{query}" in self.prompt, "{query} is required for prompt templates."
            return

        if self.name == "default":
            r"""
            Supports language models without instruction-tuning.
            """
            self.prompt = "{query}"

        elif self.name == "alpaca":
            r"""
            Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
                      https://github.com/ymcui/Chinese-LLaMA-Alpaca
            """
            self.prompt = "### Instruction:\n{query}\n\n### Response:\n"

        elif self.name == "baichuan":
            r"""
            Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
            """
            self.prompt = "<reserved_102>{query}<reserved_103>"

        elif self.name == "intern":
            r"""
            Supports: https://huggingface.co/internlm/internlm-chat-7b
            """
            self.prompt = "<|User|>:{query}<eoh>\n<|Bot|>:"

        elif self.name == "vicuna":
            r"""
            Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
                      https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
            """
            self.prompt = "USER: {query} ASSISTANT: "

        elif self.name == "ziya":
            r"""
            Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
            """
            self.prompt = "<human>:{query}\n<bot>:"

        else:
            raise NotImplementedError

    def get_prompt(self, query: str) -> str:
        return self.prompt.format(query=query)
