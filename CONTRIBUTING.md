# Contributing to PDG MCP Server

Thanks for your interest in contributing to the PDG MCP Server! This project provides easy access to particle physics data for researchers and students.

## Quick Start

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/yourusername/pdg-mcp-server
   cd pdg-mcp-server
   ```
3. **Install** dependencies:
   ```bash
   ./docker-run.sh build  # Docker method
   # OR
   pip install -r requirements.txt  # Direct method
   ```
4. **Test** your setup:
   ```bash
   ./docker-run.sh test
   ```

## Development Workflow

### Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Add new tools in `pdg_mcp_server.py`
   - Update CLI wrapper in `pdg_cli.py`
   - Add tests in `test_pdg_server.py`
   - Update documentation in `README.md`

3. **Test thoroughly**:
   ```bash
       ./docker-run.sh test           # Run full test suite
    ./pdg search --query "e-"      # Test CLI
    python examples.py             # Test examples
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "Add feature: your description"
   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request** on GitHub

### Code Style

- Follow **PEP 8** for Python code
- Use **meaningful variable names**
- Add **docstrings** for all functions
- Keep functions **focused and small**
- Add **comments** for complex logic

### Testing

- Test new features with multiple particles
- Verify both successful cases and error handling
- Test Docker and direct installation methods
   - Update `examples.py` with new examples

## Types of Contributions

### 🔧 Bug Fixes
- Fix particle name recognition issues
- Resolve Docker build problems
- Correct data formatting errors

### ✨ New Features
- Add new PDG tools (quantum numbers, cross-sections, etc.)
- Improve search functionality
- Add new CLI commands
- Better error messages

### 📚 Documentation
- Improve README examples
- Add more usage scenarios
- Better installation instructions
- API documentation

### 🧪 Testing
- Add test cases for edge cases
- Test with different particle types
- Integration testing
- Performance testing

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions about particle physics or usage
- **PDG Documentation**: https://pdgapi.lbl.gov/doc/

## Code of Conduct

- Be respectful and inclusive
- Help newcomers to particle physics
- Share knowledge and learn together
- Focus on scientific accuracy

## Technical Notes

### Adding New Tools

To add a new MCP tool:

1. **Define the tool** in `handle_list_tools()`:
   ```python
   types.Tool(
       name="your_tool_name",
       description="What it does",
       inputSchema={...}
   )
   ```

2. **Implement the logic** in `handle_call_tool()`:
   ```python
   elif name == "your_tool_name":
       # Implementation here
       return [types.TextContent(...)]
   ```

3. **Add CLI wrapper** in `pdg_cli.py`
4. **Update help** in README.md
5. **Add tests** in `test_pdg_server.py`

### PDG API Guidelines

- Use proper particle naming conventions
- Handle PDG errors gracefully
- Respect rate limits (if any)
- Cite data sources appropriately

Thanks for contributing to particle physics research! 🔬⚛️ 