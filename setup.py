from setuptools import setup, find_packages

setup(
    name="llm-bridge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "instructor==1.4.1",
        "openai==1.42.0",
        "anthropic==0.34.1",
        "pydantic==2.8.2",
        "tiktoken==0.7.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-bridge",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
