#!/usr/bin/env python3
"""
Setup script for PDG MCP Server
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="pdg-mcp-server",
    version="1.0.0",
    author="uzerone, bee4come",
    author_email="",
    description="Model Context Protocol server for Particle Data Group (PDG) data access",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uzerone/pdg-mcp-server",
    py_modules=["pdg_mcp_server"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pdg-mcp-server=pdg_mcp_server:run_server",
        ],
    },
    keywords="particle physics, PDG, MCP, research, data",
    project_urls={
        "Bug Reports": "https://github.com/uzerone/pdg-mcp-server/issues",
        "Source": "https://github.com/uzerone/pdg-mcp-server",
        "PDG Website": "https://pdg.lbl.gov/",
        "PDG API Docs": "https://pdgapi.lbl.gov/doc/",
    },
)
