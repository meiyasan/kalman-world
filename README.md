# Python Project Template

This repository provides a comprehensive template for Python projects, focusing on modern development practices using `pyproject.toml`, Docker integration, tests, and package distribution.


## Features

### Project Structure
- **`src/`**: Contains the main Python package code. Using `src/` ensures proper isolation and avoids common import issues.
- **`pyproject.toml`**: Manages dependencies, build tools, and configuration.
- **`.gitlab-ci.yml`**: Pre-configured CI/CD pipeline for automated testing and deployment.
- **`Makefile`**: Simplifies tasks:
  - `make install`: Installs dependencies.
  - `make test`: Runs tests with `pytest`.
  - `make lint`: Lints code with `flake8`.
  - `make format`: Formats code using `black`.
  - `make docker-build`: Builds a Docker image.
  - `make docker-run`: Runs the project in a Docker container.
- **Doxygen Documentation**: Generates a responsive webpage (using Doxygen Awesome) including versioning and additional documentation information based on markdown files located in `./docs`.
- **Docker Integration**: Dockerfile provided for custom image creation, facilitating CI/CD workflows.
- **GitLab CI/CD**: GitLab CI/CD templates to set up the environment, compile, and deploy library documentation through a dedicated pipeline.

## Python Package Distribution

The `src/` directory is used to store the main package, following best practices for Python projects. This approach ensures that imports work correctly and keeps your package cleanly separated from other project files. The structure is designed for easy distribution through package managers like PyPI.

### Documentation

Generate documentation using Doxygen's `make docs`. The `docs` directory contains Markdown files instrumental in generating comprehensive Doxygen documentation, available post-compilation in `./public/share/docs`. Serve this directory using HTTP servers like NGINX or Apache2.

### GitLab CI/CD and Pages Integration

Seamlessly integrate the project with GitLab CI/CD for automated testing, building, and deployment. GitLab Pages are automatically generated upon successful pipeline completion and tag publication, configurable via the GitLab repository settings post the initial successful pipeline run.

The CI/CD workflow executes on a GitLab instance, compiling the library within a Docker container, conducting comprehensive tests, and deploying documentation upon creating a new tag.

### Docker Integration

A complete Dockerfile is included to containerize the application using `Dockerfile` and `Dockerfile.packages`, ensuring consistency across environments.
  - **Build**: Use `make docker` to create a Docker image.
  - **Run**: Use `make docker-exec` to execute the project in a Docker container.

The Dockerfile is also usd as needed during CI/CD to run tests & build. 

## Usage

1. **Clone the Repository**:

    ```bash
    git clone --recursive https://git.ligo.org/kagra/templates/pyproject-template mypyproject
    cd mypyproject
    ```

2. **Bootstrap the Project**:

    Make sure to install conda, first depending on your operating system and architecture. (See [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for conda installation)
    Run the following command to prepare environment, compile, and check distribution of your python module
    ```bash
    make all
    ```

3. **Building and Testing**:

    Execute tests to ensure library functionality:
    ```bash
    make tests
    make build
    ```

4. **Generating Documentation**:

    Generate comprehensive documentation, including versioning and additional Markdown information:

    ```bash
    make docs
    ```

    Access the generated documentation from `./public/share/docs` in a web browser.

## Contributions and Support

Contributions, bug reports, and feature requests are encouraged! Refer to CONTRIBUTING.md for detailed guidelines.
