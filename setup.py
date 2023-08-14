from setuptools import setup

setup(
    name ="coreax",
    version = "0.1.0",
    description = "Jax coreset algorithms.",
    author = "GCHQ",
    packages = ["coreax"],
    install_requires = ["jax", 
                        "numpy", 
                        "jaxopt", 
                        "scikit-learn", 
                        "matplotlib", 
                        "imageio", 
                        "typing-extensions", 
                        "opencv-python",
                        "optax",
                        "flax"]
)