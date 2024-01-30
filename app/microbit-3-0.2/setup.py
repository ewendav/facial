import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = "0.2"

setuptools.setup(
    name="microbit-3",
    version=version,
    author="Phoneguytech",
    description="Microbit for Python 3",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    install_requires=["pyserial"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
