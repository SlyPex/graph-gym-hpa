from setuptools import setup , find_packages

setup(
    name="graph_gym_hpa",
    version="0.0.1",
    author="Fellah and meharzi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["gymnasium", "numpy", "keras","torch","stable-baselines3"]  
    )
