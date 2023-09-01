from setuptools import setup

setup(
    name="coreax",
    version="0.1.0",
    description="Jax coreset algorithms.",
    author="GCHQ",
    packages=["coreax"],
    install_requires=[
        "flax",
        "jax",
        "jaxopt",
        "optax",
        "scikit-learn",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "black",
            "imageio",
            "isort",
            "matplotlib",
            "numpy",
            "opencv-python",
            "pyqt5",
            "pytest",
            "scipy",
        ],
    },
)
