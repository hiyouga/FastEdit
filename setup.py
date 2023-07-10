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
        name="pyfastedit",
        version=get_version(),
        author="hiyouga",
        author_email="hiyouga" "@" "buaa.edu.cn",
        description="Editing large language models within 10 seconds",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["LLM", "ChatGPT", "transformer", "pytorch", "deep learning"],
        license="Apache 2.0 License",
        url="https://github.com/hiyouga/FastEdit",
        packages=find_packages(),
        python_requires=">=3.8.0",
        install_requires=[
            "torch>=1.13.1",
            "transformers>=4.29.1",
            "datasets>=2.12.0",
            "accelerate>=0.19.0",
            "sentencepiece",
            "fire"
        ],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ]
    )


if __name__ == "__main__":
    main()
