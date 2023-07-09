from dataclasses import dataclass


@dataclass
class Template:

    name: str

    def __post_init__(self):

        if self.name == "default":
            self.prompt = "{}"
        elif self.name == "ziya":
            self.prompt = "<human>:{}\n<bot>:"
        elif self.name == "glm":
            self.prompt = "[Round 0]\n\n问：{}\n\n答："
        else:
            raise NotImplementedError

    def get_prompt(self, query: str) -> str:
        return self.prompt.format(query)
