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

## Building the project (the C++ ROOT library, libKalman — needs ROOT + cmake)
build: motd
	@if [ ! -d "$(BUILD_DIR)" ]; then MOTD=0 $(MAKE) -s configure; fi
	@$(MAKE) -sC $(BUILD_DIR) lib

## Run the C++ (libKalman) test suite under ctest. `make tests` is the Python one.
ctests: motd
	@if [ ! -d "$(BUILD_DIR)" ]; then MOTD=0 $(MAKE) -s configure; fi
	@cmake -S . -B $(BUILD_DIR) -DBUILD_TESTS=ON > /dev/null
	@$(MAKE) -sC $(BUILD_DIR) ctests

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

## Build and install libKalmanFramework, whose kernels the Python engines call
## (optional: without it the engines run their pure-Python path). Skips, with a
## notice, where cmake is missing so a bare CI image still runs the Python suite.
accel: motd
	@if command -v cmake >/dev/null 2>&1; then \
		MOTD=0 $(MAKE) --no-print-directory build && MOTD=0 $(MAKE) --no-print-directory install; \
	else \
		echo "cmake not found: libKalmanFramework not built, the engines run in pure Python"; \
	fi

## One-command onboarding: dependencies, optional extras, C acceleration, self-check
setup: motd
	@echo "==> python: $(PYTHON)"
	@$(PYTHON) -m pip install -q --upgrade pip
	@if [ -f requirements.txt ]; then \
		echo "==> core dependencies (requirements.txt)"; \
		$(PYTHON) -m pip install -q -r requirements.txt; \
	fi
	@echo "==> optional extras (mcp, rest, auth, remote)"
	@$(PYTHON) -m pip install -q "mcp>=1.0" "fastapi>=0.110" "uvicorn[standard]>=0.29" \
		"bcrypt>=4.0" "pyjwt>=2.0" "httpx>=0.27" \
		|| echo "    (some extras unavailable - the core works without them)"
	@echo "==> C acceleration (optional)"
	@-MOTD=0 $(MAKE) --no-print-directory accel >/dev/null 2>&1 \
		&& echo "    built" || echo "    skipped (pure-Python fallback in use)"
	@echo "==> self-check"
	@$(PYTHON) -c "import kalman; from kalman.api import catalog; \
c = catalog(); \
print(f'    {len(c[\"models\"])} models, {len(c[\"engines\"])} engines, {len(c[\"integrators\"])} integrators, {len(c[\"solvers\"])} solvers, {len(c[\"providers\"])} providers')"
	@$(PYTHON) -c "import importlib; \
mods = {'mcp': 'MCP server', 'fastapi': 'REST API', 'jwt': 'scoped auth', 'torch': 'GPU/ML'}; \
[print(f'    {label:12s} {\"ready\" if importlib.util.find_spec(m) else \"not installed\"}') for m, label in mods.items()]"
	@echo ""
	@echo "Next:  make tests        run the suite"
	@echo "       make mcp          register the MCP server with your clients"
	@echo "       make serve        run REST + MCP via docker-compose"

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
mcp-serve:
	@$(PYTHON) -m kalman.mcp

## Set up MCP end-to-end: install the SDK, register with every detected client, verify
mcp: motd
	@echo "==> installing the MCP SDK (if missing)"
	@$(PYTHON) -c "import mcp" 2>/dev/null || $(PYTHON) -m pip install -q "mcp>=1.0"
	@echo "==> registering the 'Kalman-AI' server with detected clients"
	@$(PYTHON) tools/mcp_install.py $(or $(CLIENTS),--all) $(if $(TOKEN),--token "$(TOKEN)")
	@echo ""
	@echo "==> verifying the server starts"
	@$(PYTHON) -c "from kalman.mcp import build_server; s = build_server(); print('    OK: server builds -', type(s).__name__)"
	@echo ""
	@$(PYTHON) tools/mcp_install.py --list

## Register the Kalman-AI MCP server with your MCP clients: make mcp-install [CLIENTS="cursor vscode"]
mcp-install:
	@$(PYTHON) tools/mcp_install.py $(or $(CLIENTS),--all) $(if $(TOKEN),--token "$(TOKEN)")

## List MCP clients detected on this machine and whether kalman is registered
mcp-clients:
	@$(PYTHON) tools/mcp_install.py --list

## Unregister the Kalman-AI MCP server from all clients
mcp-uninstall:
	@$(PYTHON) tools/mcp_install.py --all --remove

## Add/update an API account (username + password) in the .htpasswd store: make api-user USER=alice SCOPES="read,send_measurement"
## SCOPES is what the account may ask for when it logs in (POST /token); default: keep its grant, or "read"
api-user:
	@$(PYTHON) -m kalman.api.auth adduser --user "$(USER)" $(if $(SCOPES),--scopes "$(SCOPES)")

## Revoke every token of an account issued until now: make api-revoke USER=alice   (or TOKEN=eyJ... for one token)
## API=http://localhost:8000 (or KALMAN_API_URL) revokes through the REST API -- needed when the
## servers' registry is not on this machine, e.g. the Docker services
api-revoke:
	@$(PYTHON) -m kalman.api.auth revoke $(if $(TOKEN),--token "$(TOKEN)",--user "$(USER)") $(if $(API),--api "$(API)")

## Forget an account's failed logins (a lockout ends by itself after 5 minutes): make api-unlock USER=alice [API=...]
api-unlock:
	@$(PYTHON) -m kalman.api.auth unlock --user "$(USER)" $(if $(API),--api "$(API)")

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
	@echo "==> documentation: $$(pwd)/public/latest/index.html"
	@# Open a browser only for an interactive local build (never in CI, and
	@# `open` does not exist on Linux).
	@if [ -f "public/latest/index.html" ] && [ -z "$(CI)" ] && [ -t 1 ]; then \
		if command -v open >/dev/null 2>&1; then open public/latest/index.html; \
		elif command -v xdg-open >/dev/null 2>&1; then xdg-open public/latest/index.html; fi; \
	fi

## Build project documentation
docs: motd
	@if [ ! -d "$(BUILD_DIR)/share/doxygen" ]; then BUILDTYPE=Doc MOTD=0 $(MAKE) -s configure; fi
	@$(MAKE) -sC $(BUILD_DIR) doxygen

## Replace the public/ symlink with a real directory (CI artifacts do not follow links)
pages-materialize:
	@src="$(BUILD_DIR)/public/docs"; \
	if [ -d "$$src" ]; then \
		echo "==> materialising public/ from $$src"; \
		$(RM) -rf public && cp -RL "$$src" public; \
	fi
	@test -n "$$(ls -A public 2>/dev/null)" || { echo "! no documentation was generated" >&2; exit 1; }

## Clean up build artifacts
clean: motd
	@$(RM) -rf var/cache/* var/log/* $(BUILD_DIR)/* 2> /dev/null || true

## Clean up build artifacts and remove the environment
distclean: clean
	@conda env remove -n $(ENVNAME) 2> /dev/null || true
	@$(RM) -rf .virtualenv.yml $(BUILD_DIR) $(INSTALL_DIR) 2> /dev/null || true

%:
	@:
