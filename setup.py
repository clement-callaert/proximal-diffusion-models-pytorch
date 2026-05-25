"""Install the ``src`` package and dependencies (``pip install -e .``)."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def read_requirements(filename: str) -> list[str]:
    lines = []
    for raw in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


setup(
    name="proximal-diffusion-models-pytorch",
    version="0.1.0",
    description="Proximal Diffusion Models (ProxDM) — PyTorch research implementation",
    author="Clément Callaert",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            req
            for req in read_requirements("requirements-dev.txt")
            if not req.startswith("-r")
        ],
    },
    include_package_data=True,
)
