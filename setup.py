from setuptools import setup, find_packages

readme = open('README.md').read()

setup(
    name="numpy_torch_compile_benchmark",
    version='0.0.1',
    keywords=("numpy", "torch", "asv"),
    description="The benchmarking code between numpy and torch compiled numpy",
    long_description=readme,
    long_description_content_type="text/markdown",

    license="MIT Licence",
    url="https://github.com/vra/numpy-torch-compile-benchmark",
    author="Yunfeng Wang",
    author_email="wyf.brz@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "numpy",
        "torch",
        "packaging",
    ],
)
