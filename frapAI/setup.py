"""Setup script for FPAI (Fair and Private AI) Framework."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    # Handle version constraints
                    if ">=" in line or "<=" in line or "==" in line or "<" in line or ">" in line:
                        requirements.append(line)
                    else:
                        # Skip optional dependencies and comments
                        if not line.startswith("# ") and "#" not in line:
                            requirements.append(line)
    return requirements

# Core requirements (essential for basic functionality)
core_requirements = [
    "numpy>=1.24.0,<1.25.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "qiskit>=0.45.0",
    "qiskit-aer>=0.13.0",
    "pennylane>=0.32.0",
    "torch>=2.0.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
]

# Optional dependencies
extra_requirements = {
    "full": [
        "qiskit-algorithms>=0.2.0",
        "qiskit-machine-learning>=0.7.0",
        "pennylane-qiskit>=0.32.0",
        "cirq>=1.2.0",
        "tensorflow>=2.13.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "bokeh>=3.2.0",
        "netcal>=1.3.5",
        "statsmodels>=0.14.0",
        "joblib>=1.3.0",
        "click>=8.1.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "black>=23.7.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "isort>=5.12.0",
        "pre-commit>=3.3.0",
    ],
    "docs": [
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "nbsphinx>=0.9.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
    ],
    "gpu": [
        "torch-gpu>=2.0.0",
        "tensorflow-gpu>=2.13.0",
    ],
    "privacy": [
        "opacus>=1.4.0",
        "diffprivlib>=0.6.0",
    ],
    "fairness": [
        "fairlearn>=0.9.0",
        "aif360>=0.5.0",
    ],
    "distributed": [
        "dask>=2023.7.0",
        "ray>=2.6.0",
    ],
}

# Add 'all' option that includes everything
extra_requirements["all"] = [
    req for reqs in extra_requirements.values() for req in reqs
]

setup(
    name="fpai",
    version="0.1.0",
    author="FPAI Development Team",
    author_email="fpai@example.com",
    description="Fair and Private AI Framework for Quantum Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/fpai",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/fpai/issues",
        "Documentation": "https://fpai.readthedocs.io",
        "Source Code": "https://github.com/your-username/fpai",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8,<3.14",
    install_requires=core_requirements,
    extras_require=extra_requirements,
    include_package_data=True,
    package_data={
        "fpai": [
            "*.yaml",
            "*.yml",
            "*.json",
            "data/*.csv",
            "data/*.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "fpai-benchmark=fpai.examples.benchmark_suite:main",
            "fpai-demo=fpai.examples.basic_classification:main",
        ],
    },
    keywords=[
        "quantum computing",
        "machine learning",
        "fairness",
        "privacy",
        "quantum machine learning",
        "variational quantum classifier",
        "quantum kernel",
        "calibration",
        "qiskit",
        "pennylane",
    ],
    zip_safe=False,
)