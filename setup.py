from setuptools import setup, find_packages

setup(
    name="lh_cd_project",
    version="0.1",
    packages=find_packages(where="src"),  
    package_dir={"": "src"},
)
