from setuptools import find_packages, setup


install_requirements = [
    "torch==1.11.0",
    "numpy==1.21.2",
    "pandas==1.0.5",
    "tensorboard==2.9.0",
    "protobuf==3.20.1",
    "faiss-cpu==1.7.2",
    "omegaconf==2.2.3",
    "pyarrow==9.0.0"
]

test_requirements = [
    "flake8",
    "pytest==6.2.4",
    "tox==3.24.5",
    "mypy==0.931"
]

setup(
    name="qdrl",
    author="moscicky",
    description="Library for query-document representation learning",
    packages=[package for package in find_packages(exclude=["tests*"])],
    version="0.1.0",
    license="MIT",
    install_requires=install_requirements,
    extras_require=dict(test=test_requirements),
    test_suites="tests"
)