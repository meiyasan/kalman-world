#! /usr/bin/env python3

import os
import sys

try:
	from dotenv import load_dotenv
	dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
	load_dotenv(dotenv_path)
except:
	pass

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def normalize_project_name(project_name):
    """
    Normalize the project name to derive the module name.
    """
    normalized = project_name.replace(" ", "-").lower().replace("-", "_")
    return normalized

def find_packages(base_dir, module_name):
    """
    Recursively find all packages under the base directory
    and map them to the module namespace.
    """
    packages = []
    for root, dirs, files in os.walk(base_dir):
        if "__init__.py" in files:  # Only consider directories with __init__.py as packages
            package = root.replace(os.sep, ".").replace(base_dir, module_name, 1)
            packages.append(package)
    return packages

def create_pyproject_toml(env_vars, module_name, packages, package_dir):
    """
    Generate the content of pyproject.toml based on .env variables and detected packages.
    """
    # Check if README.md exists
    readme_entry = ""
    if os.path.exists("README.md"):
        readme_entry = 'readme = "README.md"'

    # Check if LICENSE exists
    license_entry = ""
    if os.path.exists("LICENSE"):
        license_entry = 'license = { file = "LICENSE" }'

    homepage_entry = ""
    if env_vars.get('PROJECT_HOMEPAGE_URL', None):
        homepage_entry = f"""[project.urls]
homepage = "{env_vars.get('PROJECT_HOMEPAGE_URL', '')}"
"""

    # Join packages with a newline and a comma, this will create the correct format
    packages_list = ',\n\t'.join(f'"{pkg}"' for pkg in packages)

    # Generate the TOML content
    return f"""
[project]
name = "{env_vars['PROJECT_NAME']}"
version = "{env_vars.get('PROJECT_VERSION', '1.0')}"
description = "{env_vars.get('PROJECT_DESCRIPTION', '')}"
{readme_entry}
{license_entry}

{homepage_entry}
[project.optional-dependencies]
test = ["unittest"]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {{ "{module_name}" = "{package_dir}" }}  # Escaping curly braces
packages = [
\t{packages_list}
]
"""

def print_usage():
    """
    Print the usage instructions for this script.
    """
    print("""
Usage:
    python pyproject.py [package_dir]

Arguments:
    package_dir   (Optional) The directory containing the Python packages. Defaults to './modules' if not provided.

Description:
    This script generates a pyproject.toml file based on the structure of the specified directory containing Python packages.
    It reads the PROJECT_NAME from the environment (either from .env or the current directory) and normalizes it to 
    derive the module name. The script also detects all packages in the specified directory and lists them in the 
    pyproject.toml.

Example usage:
    1. Using default modules directory:
       python pyproject.py

    2. Specifying a custom modules directory:
       python pyproject.py ./path/to/python/source/module
    """)

# Check for command-line arguments to print usage
if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
    print_usage()
    sys.exit()

# Try importing load_dotenv only if available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env if present
    env_loaded = True
except ImportError:
    print("Warning: python-dotenv is not installed. Skipping .env loading.")
    env_loaded = False

# Attempt to retrieve PROJECT_NAME from the environment (either from .env or fallback)
project_name = os.getenv("PROJECT_NAME")

# If PROJECT_NAME is not defined, use the directory name as the default
if not project_name:
    project_name = os.path.basename(os.getcwd())

# Load other optional variables with defaults
env_vars = {
    "PROJECT_NAME": project_name,
    "PROJECT_VERSION": os.getenv("PROJECT_VERSION", "1.0"),
    "PROJECT_DESCRIPTION": os.getenv("PROJECT_DESCRIPTION", ""),
    "PROJECT_HOMEPAGE_URL": os.getenv("PROJECT_HOMEPAGE_URL", ""),
}

# Normalize the PROJECT_NAME to derive the MODULE name
module_name = normalize_project_name(project_name)

# Get the modules directory from the command-line argument, default to 'modules'
# Resolve the path relative to the current directory and normalize it
package_dir = sys.argv[1] if len(sys.argv) > 1 else "modules"
package_dir = os.path.normpath(package_dir)

# Ensure the package directory exists
if not os.path.exists(package_dir):
    print(f"Warning: The specified package directory '{package_dir}' does not exist.")
    sys.exit(1)

# Specify the base directory where your project modules are located
base_dir = package_dir  # Use the provided package_dir

# Find all packages and map to module namespace
packages = find_packages(base_dir, module_name)
try:
    # Generate pyproject.toml content
    pyproject_content = create_pyproject_toml(env_vars, module_name, packages, package_dir)

    # Write the content to pyproject.toml
    with open("pyproject.toml", "w") as file:
        file.write(pyproject_content)

    print("A new pyproject.toml has been successfully generated!")
except Exception as e:
    print(f"Error: Failed to generate pyproject.toml. Details: {e}")
