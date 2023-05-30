import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="sparselabel",
    version="0.0.1",
    author="Axel Henningsson",
    author_email="nilsaxelhenningsson@gmail.com",
    description="Label 3D sparse data blocks using by finding the connected components of the related directed graph.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/AxelHenningsson/sparselabel",
    project_urls={
        "Documentation": "https://axelhenningsson.github.io/sparselabel/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8,<3.11",
    install_requires=["numpy",
                      "scipy"]
)
