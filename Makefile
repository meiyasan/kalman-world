.PHONY: $(filter-out .%, $(MAKECMDGOALS))

# Setup environment variables
ifneq (,$(wildcard .env))
include .env
export
endif

export PLATFORM  ?= linux/amd64
export CONTAINER ?= $(shell basename $(PWD) | tr ' ' '-' | tr -d '.')
ifneq (,$(NOCACHE))
    export NOCACHE = "--no-cache"
endif

ifndef ENVNAME
  $(error ENVNAME is not defined. Please set it in .env or in your environment context)
endif
export MODULE    ?= $(shell echo $(PROJECT_NAME) | tr ' ' '-' | tr '[:upper:]' '[:lower:]' | tr '-' '_' || echo "unnamed")
export LINT      ?= modules tests
export DISCOVER  ?= "discover"

## Display the help message
help: usage
usage: motd
	@printf "${COLOR_TITLE_BLOCK}Project Application \"$(PROJECT_NAME)\"${COLOR_RESET}\n"
	@printf "\n"
	@printf "${COLOR_COMMENT}Usage:${COLOR_RESET}\n"
	@printf " make [target]\n\n"
	@printf "${COLOR_COMMENT}Available targets:${COLOR_RESET}\n"
	@awk 'BEGIN { inComment = 0 } \
	/^[a-zA-Z\-\_0-9\@]+:/ { \
		if (inComment) { \
			helpCommand = substr($$1, 0, index($$1, ":")); \
			printf " ${COLOR_INFO}%-16s${COLOR_RESET} %s\n", helpCommand, helpMessage; \
			inComment = 0; \
		} \
	} \
	/^##/ { \
		helpMessage = substr($$0, 3); \
		inComment = 1; \
	} \
	' $(MAKEFILE_LIST)

## Complete procedure to build and install
all: motd env tests docs build install
motd:
ifneq ($(MOTD),0)
	@if [ -s "share/motd" ] && [ ! -z "$$(which envsubst)" ]; then echo "$$(env | envsubst < share/motd)"; fi
endif

## Create a Conda environment from environment.yml
env: motd
	@echo "Checking for environment.. "
	@$(MAKE) --no-print-directory .virtualenv

.virtualenv: environment*.yml requirements.txt
	@$(MAKE) .envcheck

	@if [ -f "environment.yml" ]; then \
		if conda env list | grep -q $(ENVNAME); then \
			echo "Updating existing environment \`$(ENVNAME)\` with \`environment.yml\`"; \
			conda env update -n $(ENVNAME) -f environment.yml; \
		else \
			echo "Creating new environment \`$(ENVNAME)\` with \`environment.yml\`"; \
			conda env create -n $(ENVNAME) -f environment.yml; \
		fi; \
	else \
		echo "File \`environment.yml\` not found, skipping."; \
	fi

	@if [ -f "environment.$(shell arch).yml" ]; then \
		if conda env list | grep -q $(ENVNAME); then \
			echo "Updating existing environment \`$(ENVNAME)\` with \`environment.$(shell arch).yml\`"; \
			conda env update -n $(ENVNAME) -f environment.$(shell arch).yml; \
		else \
			echo "Creating new environment \`$(ENVNAME)\` with \`environment.$(shell arch).yml\`"; \
			conda env create -n $(ENVNAME) -f environment.$(shell arch).yml; \
		fi; \
	else \
		echo "File \`environment.$(shell arch).yml\` not found, skipping."; \
	fi

	@if [ -f "requirements.txt" ]; then \
		if ! grep -q "requirements.txt" "environment.yml" "environment.$(shell arch).yml" 2>/dev/null; then \
			echo "Installing additional Python packages from \`requirements.txt\`"; \
			conda run -n $(ENVNAME) pip install -r requirements.txt; \
		else \
			echo "\`requirements.txt\` is already referenced in environment files, skipping Python package installation."; \
		fi \
	else \
		echo "No \`requirements.txt\` found, skipping Python package installation."; \
	fi

	@if [ -f "environment.$(shell arch).yml" ]; then \
		if conda env list | grep -q $(ENVNAME); then \
			echo "Updating existing environment \`$(ENVNAME)\` with \`environment.$(shell arch).yml\`"; \
			conda env update -n $(ENVNAME) -f environment.$(shell arch).yml; \
		else \
			echo "Creating new environment \`$(ENVNAME)\` with \`environment.$(shell arch).yml\`"; \
			conda env create -n $(ENVNAME) -f environment.$(shell arch).yml; \
		fi \
	else \
		echo "File \`environment.$(shell arch).yml\` not found, skipping."; \
	fi

	@if [ -f "requirements.txt" ]; then \
		if ! grep -q "requirements.txt" environment.yml environment.$(shell arch).yml 2>/dev/null; then \
			echo "Installing additional Python packages from \`requirements.txt\`"; \
			conda run -n $(ENVNAME) pip install -r requirements.txt; \
		else \
			echo "\`requirements.txt\` is already referenced in environment files, skipping Python package installation."; \
		fi \
	else \
		echo "No \`requirements.txt\` found, skipping Python package installation."; \
	fi

	@touch .virtualenv

.envcheck:
	@. $$(conda info --base)/etc/profile.d/conda.sh
	@command -v conda >/dev/null 2>&1 || { echo "Error: conda is not installed. Please install conda to use this command."; exit 1; }
	@command -v mamba >/dev/null 2>&1 || { conda activate base && conda install mamba; }
	@command -v mamba >/dev/null 2>&1 || { echo "Error: mamba is not installed. Please install mamba to use this command."; exit 1; }

## Test the project
tests: motd
	@python -m unittest "$(DISCOVER)"; ec=$$?; [ $$ec -eq 0 ] || [ $$ec -eq 5 ] || exit $$ec

## Vendor dependencies
vendor: motd
	@$(MAKE) -sC vendor

## Build project documentation and open in browser
pages: docs
	@cd var/build/public/docs; ln -sf $$(git describe --tags 2> /dev/null || git rev-parse --short HEAD)/ latest
	@cp var/build/index.html var/build/public/docs/
	@find var/build/public/docs -mindepth 1 -maxdepth 1 -type d | rev | cut -d'/' -f 1 | rev | sort > var/build/public/docs/metadata.txt
	@rsync -a --delete ./var/build/public/ var/www 2> /dev/null || true
	@if [ -f "public/latest/index.html" ]; then open public/latest/index.html; fi

## Build project documentation
docs: motd
	@MOTD=0 $(MAKE) -s env
	@mkdir -p var/build/public/docs var/www/docs && ln -sf var/www/docs/ ./public
	@cd var/build; if [ ! -f "Makefile" ]; then cmake ../../share/doxygen; fi
	@doxygen -u var/build/Doxyfile && doxygen var/build/Doxyfile;

## Build the python package and distribute it
modules: motd
	@mkdir -p var/install/lib var/build
	@ln -sfnT ../modules tools/$(MODULE)
	@ln -sfnT ../modules tests/$(MODULE)
	@ln -sfnT ../../modules var/build/modules
	@ln -sf   ../../pyproject.toml var/build
	@export EGG_CACHE=var/build/.eggs; \
	SITEPACKAGE="var/install/lib/python-$$(python --version | awk '{print $$2}' | cut -d '.' -f 1,2)/site-packages"; \
	 cd var/build && python -m build --skip-dependency-check --no-isolation --outdir ../../$$SITEPACKAGE ; \
	 [ $$? -eq 0 ] && echo "Distributed package is now available in \`./$$SITEPACKAGE\`" || exit 1

## Install the python package in the environment
pip: motd
	@SITEPACKAGE="var/install/lib/python-$$(python --version | awk '{print $$2}' | cut -d '.' -f 1,2)/site-packages"; \
	 mkdir -p $$SITEPACKAGE && pip install $$SITEPACKAGE/$(MODULE)-$(PROJECT_VERSION).tar.gz;

## Run the linter for python package
lint: motd
	@flake8 $(LINT)

## Format the code using black
format: motd 
	@read -p "Make sure to commit your code first; Do you want to modify your code using black? (yes/no): " response; \
	if [ "$$response" = "yes" ]; then \
		black $(LINT)
	else \
		echo "Exiting..."; \
		exit 1; \
	fi

## Build the Docker image
image: motd
	@docker buildx build --platform=$(PLATFORM) -t $$(basename $(PWD)) $(NOCACHE) .

## Prepare the container (including warm up)
container:
	@$(MAKE) stop
	@docker compose -f docker-compose.yml up --build

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

## Clean up build artifacts
clean: motd
	@$(RM) -rf var/cache/* var/log/* var/build/* 2> /dev/null || true

## Clean up build artifacts and remove the environment
distclean: clean
	@conda env remove -n $(ENVNAME) 2> /dev/null || true
	@$(RM) -rf .virtualenv var/install var/build 2> /dev/null || true

%:
	@:
