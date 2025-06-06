# Release Checklist - One-Click Installation Feature

## Files Added/Modified

### New Files
- [ ] `claude-desktop-quick.json` - Simple MCP configuration for Claude Desktop
- [ ] `mcp-server.json` - Basic MCP server configuration
- [ ] `mcp-config-easy.json` - Comprehensive configuration with multiple methods
- [ ] `QUICK_START.md` - Detailed quick start guide
- [ ] `test_quick_install.py` - Test script for installation methods
- [ ] `.github/workflows/test-quick-install.yml` - GitHub Actions workflow
- [ ] `RELEASE_CHECKLIST.md` - This file

### Modified Files
- [ ] `README.md` - Added one-click installation section at the top

## Testing

### Local Tests
- ✅ `test_modular.py` - All tests pass
- ✅ JSON configurations are valid
- ⚠️  `uvx` installation test (timeout expected without full installation)
- ❌ `pipx` not installed locally (optional)

### GitHub Actions Tests
- [ ] CI workflow should pass for existing tests
- [ ] New quick-install workflow should run

## Installation Methods Supported

1. **uvx** (recommended)
   - Command: `uvx --from git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git pp-mcp-server`
   
2. **pipx**
   - Command: `pipx run --spec git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git pp-mcp-server`

3. **Direct pip install**
   - Command: `pip install git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git`

## Pre-Push Checklist

- [x] All new files created
- [x] README updated with prominent one-click install section
- [x] Test scripts work locally
- [x] JSON configurations are valid
- [x] QUICK_START guide is comprehensive
- [ ] GitHub Actions workflows are ready

## Git Commands to Push

```bash
# Add all new files
git add claude-desktop-quick.json mcp-server.json mcp-config-easy.json
git add QUICK_START.md test_quick_install.py RELEASE_CHECKLIST.md
git add .github/workflows/test-quick-install.yml

# Add modified files
git add README.md

# Commit
git commit -m "feat: Add one-click installation support for MCP

- Add simple JSON configurations for Claude Desktop
- Create comprehensive QUICK_START.md guide
- Add installation test scripts
- Update README with prominent easy install section
- Add GitHub Actions workflow for testing quick install
- Support uvx, pipx, and direct installation methods"

# Push to repository
git push origin main
```

## Post-Push Tasks

- [ ] Verify GitHub Actions run successfully
- [ ] Test installation from GitHub using uvx
- [ ] Create a GitHub Release highlighting the new feature
- [ ] Update any documentation wikis

## Feature Benefits

✅ **No local installation required** - Users can use the server without cloning
✅ **Multiple installation methods** - uvx, pipx, or direct pip
✅ **Simple JSON configuration** - Just copy and paste
✅ **Automatic dependency handling** - PDG and other deps installed automatically
✅ **Cross-platform support** - Works on Windows, Mac, and Linux 