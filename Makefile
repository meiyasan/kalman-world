.PHONY: $(filter-out .%, $(MAKECMDGOALS)) .virtualenv*

# Setup environment variables
ifneq (,$(wildcard .env))
include .env
export
endif

# Create an outdated .virtualenv file if it doesn't exist, with timestamp set to epoch (to prevent issue at docker stage)
$(shell touch .bash_history .virtualenv.yml)

export PLATFORM  ?= linux/amd64
export CONTAINER ?= $(shell basename $(PWD) | tr ' ' '-')
ifneq (,$(NOCACHE))
    export NOCACHE = "--no-cache"
endif
export CONDA_ROOT ?= $(shell conda info --base 2>/dev/null || echo "$(HOME)/.conda")

BUILD_DIR   := $(shell if [ -f "/.dockerenv" ]; then echo "var/container/.build"; else echo "var/build"; fi)
INSTALL_DIR := $(shell if [ -f "/.dockerenv" ]; then echo "var/container/.install"; else echo "var/install"; fi)

# Python interpreter used by the test/coverage/linter targets. Prefer a local
# virtualenv (.venv), then a conda env named $(ENVNAME), then the system python3.
# Override explicitly with `make tests PYTHON=...` if needed.
ifndef PYTHON
PYTHON := $(shell \
	if [ -x ".venv/bin/python" ]; then echo ".venv/bin/python"; \
	elif command -v conda >/dev/null 2>&1 && conda env list | grep -q "$(ENVNAME)"; then echo "conda run -n $(ENVNAME) python"; \
	else echo "python3"; fi)
endif

## Display the help message
help: usage
usage: motd
	@printf "${COLOR_TITLE_BLOCK}Project Application \"$(PROJECT_NAME)\"${COLOR_RESET}\n"
	@printf "\n"
	@printf "${COLOR_COMMENT}Usage:${COLOR_RESET}\n"
	@printf "\tmake [target]\n\n"
	@printf "${COLOR_COMMENT}Available targets:${COLOR_RESET}\n"
	@awk 'BEGIN { inComment = 0 } \
	/^[a-zA-Z\-\_0-9\@]+:/ { \
		if (inComment) { \
			helpCommand = substr($$1, 0, index($$1, ":")); \
			printf "\t${COLOR_INFO}%-16s${COLOR_RESET} %s\n", helpCommand, helpMessage; \
			inComment = 0; \
		} \
	} \
	/^##/ { \
		helpMessage = substr($$0, 3); \
		inComment = 1; \
	} \
	' $(MAKEFILE_LIST)

## Complete procedure to build and install
all: motd env vendor build install
motd:
ifneq ($(MOTD),0)
	@if [ -s "share/motd" ] && [ ! -z "$$(which envsubst)" ]; then echo "$$(env | envsubst < share/motd)"; fi
endif

## Download the necessary PDF sets
lhapdf:
#	@python3 lhapdf5-getdata #SMRS ?
	@lhapdf install GRVPI0 GRVPI1
	@lhapdf install CT14nlo

legacy: motd
	@echo "Checking for legacy environment.. "
	@if conda env list | grep -q $(ENVLEGACY); then \
			echo "Updating existing environment \`$(ENVLEGACY)\` with \`environment.legacy.yml\`"; \
			conda env update -n $(ENVLEGACY) -f environment.legacy.yml -v; \
		else \
			echo "Creating new environment \`$(ENVLEGACY)\` with \`environment.legacy.yml\`"; \
			conda env create -n $(ENVLEGACY) -f environment.legacy.yml -v; \
		fi

## Setup conda
conda:
	@test -n "$(CONDA_ROOT)" || { echo "ERROR: CONDA_ROOT not set"; exit 1; }
	@if [ -f "$(CONDA_ROOT)/bin/conda" ]; then \
		echo "conda found at $(CONDA_ROOT)"; \
	elif command -v conda >/dev/null 2>&1; then \
		echo "conda found at $$(conda info --base)"; \
	else \
		url=$$(uname -s | grep -q Darwin && echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-$$(uname -m).sh" || echo "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$$(uname -m).sh"); \
		tmp=$$(mktemp).sh; curl -fsSL "$$url" -o "$$tmp"; chmod +x "$$tmp"; \
		"$$tmp" -b -p "$(CONDA_ROOT)"; rm -f "$$tmp"; \
		conda tos accept --override-channels -c https://repo.anaconda.com/pkgs/main -c https://repo.anaconda.com/pkgs/r; \
	fi
	
	@. $(CONDA_ROOT)/etc/profile.d/conda.sh && \
	CONDA_CURRENT_VERSION="$$(conda --version 2>/dev/null | awk '{print $$2}')" && \
	if [ -n "$(CONDA_VERSION)" ]; then \
		if [ -z "$$CONDA_CURRENT_VERSION" ]; then \
			echo "conda command not available after install; skipping version enforcement"; \
		elif [ "$$CONDA_CURRENT_VERSION" != "$(CONDA_VERSION)" ]; then \
			echo "adjusting conda from $$CONDA_CURRENT_VERSION to $(CONDA_VERSION) (based on CONDA_VERSION)"; \
			conda install -n base -y "conda=$(CONDA_VERSION)"; \
		else \
			echo "conda is set to ($$CONDA_CURRENT_VERSION)"; \
		fi; \
	else \
		echo "skipping conda version enforcement"; \
	fi
	@echo ""
	@echo "run the following command to activate conda:"
	@echo "     source $(CONDA_ROOT)/etc/profile.d/conda.sh"
	@echo "     conda activate base"
	@echo ""

## Create a Conda environment from environment.yml
env: motd
	@echo "Checking for environment.. "
	@$(MAKE) --no-print-directory .virtualenv.yml

.envcheck:
	@. $(CONDA_ROOT)/etc/profile.d/conda.sh && \
		(command -v conda-merge >/dev/null 2>&1 || python -m pip install --quiet conda-merge) && $(MAKE) conda

.virtualenv.yml: environment*.yml requirements.txt

	@$(MAKE) .envcheck

	@envfiles=""; \
	[ -f "environment.yml" ] && envfiles="environment.yml"; \
	[ -f "environment.$(shell arch).yml" ] && envfiles="$$envfiles environment.$(shell arch).yml"; \
	if [ -n "$$envfiles" ]; then \
		filecount=$$(echo $$envfiles | wc -w | tr -d ' '); \
		if [ "$$filecount" -gt 1 ]; then \
			echo "Merging $$envfiles (for $(shell arch))"; \
			conda-merge $$envfiles > .virtualenv.yml; \
		else \
			echo "Using single $$envfiles (for $(shell arch))"; \
			cp $$envfiles .virtualenv.yml; \
		fi; \
		if conda env list | grep -q $(ENVNAME); then \
			echo "Updating environment \`$(ENVNAME)\`"; \
			conda env update -n $(ENVNAME) -f .virtualenv.yml; \
		else \
			echo "Creating environment \`$(ENVNAME)\`"; \
			conda env create -n $(ENVNAME) -f .virtualenv.yml; \
		fi; \
	else \
		echo "No environment files found"; \
		touch .virtualenv.yml; \
	fi

	@if [ -f "requirements.txt" ]; then \
		if ! grep -q "requirements.txt" ".virtualenv.yml" 2>/dev/null; then \
			echo "Installing additional Python packages from \`requirements.txt\`"; \
			conda run -n $(ENVNAME) pip install -r requirements.txt; \
		else \
			echo "\`requirements.txt\` is already referenced in environment files, skipping Python package installation."; \
		fi \
	else \
		echo "No \`requirements.txt\` found, skipping Python package installation."; \
	fi

## Run a single command inside the container (usage: make console "command")
console:
	@MOTD=0 $(MAKE) start
	@CONTAINER_ID=$$(docker ps -aqf ancestor=$${CONTAINER}-container) && \
	 if [ ! -z "$$CONTAINER_ID" ]; then docker exec -it $$CONTAINER_ID /bin/bash -c "$(filter-out $@,$(MAKECMDGOALS))"; else echo "No running container found, please check \`docker ps\`"; fi

## Building with debug symbols
debug: motd
	@MOTD=0 BUILDTYPE=Debug $(MAKE) -s configure

## Building in release mode
release: motd
	@MOTD=0 BUILDTYPE=Release $(MAKE) -s configure

## Configure the build environment and build
configure: motd
	@MOTD=0 ./bootstrap.sh $(BUILDTYPE)

## Building the project
build: motd
	@if [ ! -d "$(BUILD_DIR)" ]; then MOTD=0 $(MAKE) -s configure; fi
	@$(MAKE) -sC $(BUILD_DIR) lib

## Vendor dependencies
vendor: motd
	@$(MAKE) -sC vendor

## Compile the tools
tools: motd
	@if [ ! -d "$(BUILD_DIR)" ]; then BUILDTYPE=Release MOTD=0 $(MAKE) -s configure; fi
	@$(MAKE) -sC $(BUILD_DIR) tools

## Compile the examples
examples: motd
	@if [ ! -d "$(BUILD_DIR)" ]; then BUILDTYPE=Release MOTD=0 $(MAKE) -s configure; fi
	@$(MAKE) -sC $(BUILD_DIR) examples

## Build the optional C acceleration extension for the Kalman engines
accel: motd
	@$(PYTHON) -m modules.framework.native.cffi

## Run the Python unit-test suite with a per-module coverage summary
tests: motd
	@FORCE_COLOR=1 $(PYTHON) -m pytest tests/ --tb=short --color=yes \
		--cov=modules --cov-report=term
	@echo ""
	@$(PYTHON) tests/bench_native.py

## Run the unit tests, then list every uncovered function by name
coverage: motd
	@FORCE_COLOR=1 $(PYTHON) -m pytest tests/ --tb=short --color=yes \
		--cov=modules --cov-report=term -q
	@$(PYTHON) share/coverage.py

## Lint the Python sources with ruff (correctness-focused; see [tool.ruff] in pyproject)
linter: motd
	@$(PYTHON) -m ruff check modules/ tests/

## Install the project in the final directory
install: motd
	@if [ ! -d "$(BUILD_DIR)" ]; then BUILDTYPE=Release MOTD=0 $(MAKE) -s configure; fi
	@$(MAKE) -sC $(BUILD_DIR) install

## Serve the REST API + MCP server via docker-compose
serve: motd
	@docker compose up -d rest mcp

## Run the REST API locally (uvicorn, default 0.0.0.0:8000)
rest:
	@$(PYTHON) -m kalman.rest

## Run the MCP server locally (stdio)
mcp:
	@$(PYTHON) -m kalman.mcp

## Register the Kalman MCP server with your MCP clients: make mcp-install [CLIENTS="cursor vscode"]
mcp-install:
	@$(PYTHON) tools/mcp_install.py $(or $(CLIENTS),--all) $(if $(TOKEN),--token "$(TOKEN)")

## List MCP clients detected on this machine and whether kalman is registered
mcp-clients:
	@$(PYTHON) tools/mcp_install.py --list

## Unregister the Kalman MCP server from all clients
mcp-uninstall:
	@$(PYTHON) tools/mcp_install.py --all --remove

## Add/update an API account (username + password) in the .htpasswd store: make api-user USER=alice
api-user:
	@$(PYTHON) -m kalman.api.auth adduser --user "$(USER)"

## Mint a scoped JWT for an API account: make api-token USER=alice SCOPES="read,send_measurement"
api-token:
	@$(PYTHON) -m kalman.api.auth token --user "$(USER)" --scopes "$(or $(SCOPES),read)" --exp "$(or $(EXP),24)"

## Mint a full-scope admin JWT: make api-admin-token USER=root
api-admin-token:
	@$(PYTHON) -m kalman.api.auth token --user "$(USER)" --scopes "admin" --exp "$(or $(EXP),24)"

## Build the Docker image
image: motd
	@docker buildx build --platform=$(PLATFORM) -t $$(basename $(PWD)) $(NOCACHE) .

## Prepare the container (including warm up)
container:
	@$(MAKE) stop
	@docker compose -f docker-compose.yml up --build --force-recreate --remove-orphans

## Start the container
start: motd
	@docker compose up -d

## Stop the container
stop: motd
	@docker compose down

## Restart the container
restart: motd
	@MOTD=0 $(MAKE) stop
	@MOTD=0 $(MAKE) start

## Run in a shell inside the container
shell: motd
	@MOTD=0 $(MAKE) start
	@CONTAINER_ID=$$(docker ps -aqf ancestor=$${CONTAINER}-container) && \
	 if [ ! -z "$$CONTAINER_ID" ]; then docker exec -it $$CONTAINER_ID /bin/bash; else echo "No running container found, please check \`docker ps\`"; fi

## Build project documentation and open in browser
pages: motd
	@if [ ! -d "$(BUILD_DIR)/share/doxygen" ]; then BUILDTYPE=Doc MOTD=0 $(MAKE) -s configure; fi
	@$(MAKE) -sC $(BUILD_DIR) pages
	@if [ -f "public/latest/index.html" ]; then open public/latest/index.html; fi

## Build project documentation
docs: motd
	@if [ ! -d "$(BUILD_DIR)/share/doxygen" ]; then BUILDTYPE=Doc MOTD=0 $(MAKE) -s configure; fi
	@$(MAKE) -sC $(BUILD_DIR) doxygen

## Clean up build artifacts
clean: motd
	@$(RM) -rf var/cache/* var/log/* $(BUILD_DIR)/* 2> /dev/null || true

## Clean up build artifacts and remove the environment
distclean: clean
	@conda env remove -n $(ENVNAME) 2> /dev/null || true
	@$(RM) -rf .virtualenv.yml $(BUILD_DIR) $(INSTALL_DIR) 2> /dev/null || true

%:
	@:
