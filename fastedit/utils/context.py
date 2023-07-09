import os
import json
from typing import List


def get_context_templates() -> List[str]:

    dir_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(dir_path, "context.json"), "r", encoding="utf-8") as f:
        return json.load(f)
