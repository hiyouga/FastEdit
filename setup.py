import os
import re
from setuptools import setup, find_packages


def get_version():
    with open(os.path.join("fastedit", "__init__.py"), "r", encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{0}\W*=\W*\"([^\"]+)\"".format("__version__")
        version, = re.findall(pattern, file_content)
        return version


def main():

    setup(
        name="fastedit",
        version=get_version(),
        author="hiyouga",
        author_email="hiyouga" "@" "buaa.edu.cn",
        url="https://github.com/hiyouga/FastEdit",
        description="Editing large language models within 10 Seconds",
        license="Apache-2.0",
        packages=find_packages(include="fastedit"),
        install_requires=[
            "torch>=1.13.1",
            "transformers>=4.29.1",
            "datasets>=2.12.0",
            "accelerate>=0.19.0",
            "sentencepiece",
            "fire"
        ]
    )


if __name__ == "__main__":
    main()
