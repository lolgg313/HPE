from setuptools import setup, find_packages

setup(
    name="hpe",
    version="1.0.0",
    description="Hamid Py Engine",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="XO Aria, S.U.P.E",
    author_email="hf18950@gmail.com, hamedsheygh@gmail.com",
    license="MIT",
    url="https://github.com/xo-aria/HPE",
    project_urls={
        "Homepage": "https://github.com/xo-aria/HPE",
        "Repository": "https://github.com/xo-aria/HPE",
        "Issues": "https://github.com/xo-aria/HPE/issues",
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "customtkinter==5.2.2",
        "darkdetect==0.8.0",
        "llvmlite==0.44.0",
        "numba==0.61.2",
        "numpy==2.2.6",
        "packaging==25.0",
        "pillow==11.3.0",
        "pygame==2.6.1",
        "PyOpenGL==3.1.9",
        "pyopengltk==0.0.4",
        "toml==0.10.2",
        "trimesh==4.7.1"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Games/Entertainment :: Simulation",
        "Framework :: Pygame"
    ],
    python_requires=">=3.8",
)
