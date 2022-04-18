from setuptools import find_packages, setup


install_requirements = [
    "torch==1.9.0"
    "numpy==1.21.2",
    "pyspark==2.4.8",
    "pyspark-stubs==2.4.0",
    "pandas==1.0.5",
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