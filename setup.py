from setuptools import setup

setup(
    name="coreax",
    version="0.1.0",
    description="Jax coreset algorithms.",
    author="GCHQ",
    packages=["coreax"],
    python_requires=">=3.9",
    install_requires=[
        "jax",
        "jaxopt",
        "scikit-learn",
    ],
)
