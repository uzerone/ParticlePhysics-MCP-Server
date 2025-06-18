#!/usr/bin/env python3
"""
Setup script for ParticlePhysics MCP Server
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="ParticlePhysics-mcp-server",
    version="1.0.0",
    author="uzerone, bee4come",
    description="Model Context Protocol server for Particle Data Group (PDG) physics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uzerone/ParticlePhysics-MCP-Server",
    packages=find_packages(),
    py_modules=["pp_mcp_server", "pp_mcp_help"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pp-mcp-server=pp_mcp_server:run_server",
            "pp-mcp-help=pp_mcp_help:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/uzerone/ParticlePhysics-MCP-Server/issues",
        "Source": "https://github.com/uzerone/ParticlePhysics-MCP-Server",
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
)
