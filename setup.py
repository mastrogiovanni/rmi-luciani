import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="luciani-pkg-mastrogiovanni", # Replace with your own username
    version="0.0.1",
    author="Ludovica Luciani",
    author_email="ludovica.luciani95@gmail.com",
    description="Graph algorithms for random parcelization thesys",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mastrogiovanni/rmi-luciani",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License 2",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)