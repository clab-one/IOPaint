import setuptools
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def load_requirements():
    # FOLIO fork: 원본은 빈 줄과 주석까지 그대로 넘겨서 requirements.txt 에
    # 설명을 못 달았다.
    requires = []
    for line in Path("requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requires.append(line)
    return requires


# https://setuptools.readthedocs.io/en/latest/setuptools.html#including-data-files
setuptools.setup(
    name="folio-iopaint",
    version="1.6.0+folio.1",
    author="PanicByte",
    author_email="cwq1913@gmail.com",
    description="FOLIO fork of IOPaint: headless erase (LaMa) service, no web UI, no diffusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clab-one/IOPaint",
    packages=setuptools.find_packages("."),
    install_requires=load_requirements(),
    python_requires=">=3.10",
    entry_points={"console_scripts": ["iopaint=iopaint:entry_point"]},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
