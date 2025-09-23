# Enable color support for ls and grep

if [ -x /usr/bin/dircolors ]; then
    eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# Shortcuts for history search
export HISTSIZE=
export HISTFILESIZE=

export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'
export BASH_SILENCE_DEPRECATION_WARNING=1
if [[ $- == *i* ]]
then
        bind '"\e[A": history-search-backward'
        bind '"\e[B": history-search-forward'
        bind '"\e[5C": forward-word'
        bind '"\e[5D": backward-word'
        bind '"\e[1;5C": forward-word'
        bind '"\e[1;5D": backward-word'
fi

shopt -s checkwinsize
shopt -s direxpand 2> /dev/null

# Function to construct the prompt dynamically
function set_prompt() {
    # Check if running inside Docker by looking for /.dockerenv
    local docker=""
    if [ -f /.dockerenv ]; then
        docker="\[\e[36m\]docker:"  # Cyan color for 'on docker'
    fi

    # Check if a conda environment is active
    local conda=""
    if [[ $CONDA_DEFAULT_ENV != "" ]]; then
        conda="\[\e[90m\]($CONDA_DEFAULT_ENV)\[\e[0m\] "
    fi

    # Construct the dynamic PS1
    PS1="\[\e[95;1m\][\[\e[95;1m\]\t] \[\e[32;1m\]\u@$docker\h\[\e[0;97m\]:\[\e[0;94m\]\w\[\e[39m\] $\[\e[0;97m\] \n\[\e[0;97m\]$conda\[\e[0m\]"
}

# Set PROMPT_COMMAND to call the set_prompt function
PROMPT_COMMAND=set_prompt

# Source environment file
[[ $_ != $0 ]] && REALPATH=`dirname $(readlink -f ${BASH_SOURCE[0]})` || REALPATH=`dirname $(readlink -f $0)`
[ -f "$REALPATH/.env" ] && source $REALPATH/.env

# Enable conda environment (if found)
if [ ! -z "$(which conda)" ]; then
    
    conda env list
    ENVNAME="${ENVNAME:-base}"

    source $(conda info --base)/etc/profile.d/conda.sh
    if [[ ! -z "$(conda env list | grep "^$ENVNAME ")" ]]; then
        conda activate "$ENVNAME"
    else 
        conda activate base
    fi
fi