from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="shade-scheduler",
    version="1.0.0",
    author="[Padala sravan]",
    author_email="[padalasravan9@gmail.com]",
    description=(
        "SHADE: Scalable Heterogeneity-Aware Deep Reinforcement Learning "
        "for Edge-Cloud Task Scheduling"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/SHADE",
    packages=find_packages(exclude=["tests*", "experiments*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    keywords=[
        "deep reinforcement learning", "edge computing", "task scheduling",
        "heterogeneous computing", "IoT", "multi-objective optimization",
        "transfer learning", "hierarchical DRL"
    ],
    project_urls={
        "Paper": "https://doi.org/10.XXXX/XXXX",
        "Bug Reports": "https://github.com/[username]/SHADE/issues",
        "Source": "https://github.com/[username]/SHADE",
    },
)
