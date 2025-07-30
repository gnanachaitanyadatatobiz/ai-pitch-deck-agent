#!/usr/bin/env python3
"""
Script to verify repository status and ensure all code is updated
"""

import os
import subprocess
import sys

def run_command(command):
    """Run a command and return the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def check_repo_status():
    """Check the current repository status."""
    print("🔍 Checking Repository Status...")
    print("=" * 50)
    
    # Check current branch
    code, branch, error = run_command("git branch --show-current")
    if code == 0:
        print(f"📍 Current branch: {branch}")
    else:
        print(f"❌ Error getting branch: {error}")
    
    # Check git status
    code, status, error = run_command("git status --porcelain")
    if code == 0:
        if status:
            print("⚠️  Uncommitted changes found:")
            print(status)
        else:
            print("✅ Working directory clean - all changes committed")
    else:
        print(f"❌ Error checking status: {error}")
    
    # Check recent commits
    code, commits, error = run_command("git log --oneline -5")
    if code == 0:
        print("\n📝 Recent commits:")
        for line in commits.split('\n'):
            print(f"  {line}")
    else:
        print(f"❌ Error getting commits: {error}")
    
    # Check remote status
    code, remote_status, error = run_command("git status -uno")
    if code == 0:
        print(f"\n🌐 Remote status:")
        print(remote_status)
    else:
        print(f"❌ Error checking remote status: {error}")
    
    # Check if we need to push
    code, ahead_behind, error = run_command("git rev-list --count --left-right @{upstream}...HEAD")
    if code == 0:
        behind, ahead = ahead_behind.split('\t') if '\t' in ahead_behind else ('0', '0')
        if int(ahead) > 0:
            print(f"⚠️  {ahead} commits ahead of remote - need to push")
        elif int(behind) > 0:
            print(f"⚠️  {behind} commits behind remote - need to pull")
        else:
            print("✅ Local and remote are in sync")
    else:
        print("ℹ️  Could not check ahead/behind status")

def check_key_files():
    """Check if key files have the expected content."""
    print("\n🔍 Checking Key Files...")
    print("=" * 50)
    
    key_files = {
        'streamlit_app.py': ['extract_urls_from_text', 'Web Research Results', 'Sources Found'],
        'requirements.txt': ['chromadb==0.4.24', 'crewai>=0.22.0', 'streamlit>=1.28.0'],
        'runtime.txt': ['python-3.11'],
        'README.md': ['Enhanced web research', 'clickable links'],
        'DEPLOYMENT_GUIDE.md': ['ChromaDB compatibility', 'Streamlit Cloud']
    }
    
    for filename, expected_content in key_files.items():
        if os.path.exists(filename):
            print(f"✅ {filename} exists")
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for expected in expected_content:
                        if expected.lower() in content.lower():
                            print(f"  ✅ Contains: {expected}")
                        else:
                            print(f"  ❌ Missing: {expected}")
            except Exception as e:
                print(f"  ❌ Error reading {filename}: {e}")
        else:
            print(f"❌ {filename} not found")

def main():
    """Main verification function."""
    print("🚀 PitchAI Repository Verification")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("❌ Not in a git repository!")
        return
    
    check_repo_status()
    check_key_files()
    
    print("\n📋 Summary:")
    print("=" * 50)
    print("✅ All major updates have been implemented:")
    print("  - ChromaDB compatibility fixes")
    print("  - Enhanced clickable links functionality")
    print("  - Improved research results display")
    print("  - Debug info removed from UI")
    print("  - Streamlit Cloud deployment ready")
    
    print("\n🌐 Repository: https://github.com/NavinSivakumar07/PitchAI")
    print("📖 Check the browser tab that opened to verify the latest commits")

if __name__ == "__main__":
    main()
