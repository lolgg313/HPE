from setuptools import setup, find_packages

setup(
    name="hpe",
    version="2.2.9",
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
    entry_points={
        "console_scripts": [
            "hpe=hpe.cli:main",
        ],
    },
    author="S.U.P.E, Alireza Enhessari",
    author_email="hamedsheygh3130011@gmail.com",
    description="Hamid Py Engine",
    license="Hamid PY Engine License",
    classifiers=[
        "License :: Other/Proprietary License",
    ],
    package_data={
        "hpe": ["code/hpe.py"],
    },
    python_requires=">=3.8",
)
