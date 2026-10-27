#!/bin/bash

# It is a good practice to wrap the entire script in a function.
# This prevents variables defined inside from leaking into the user's shell.
_setup_env() {
    # --- Step 1: Save current shell options ---
    # This is the crucial step. We save the state of `set -e` and other options.
    local original_shell_options=$(set +o)

    # It's safe to use 'set -e' inside this function now, if needed,
    # as we will restore the original settings at the end.
    # For this script, it's not strictly necessary, but it's good practice.
    set -e

    # --- Step 2: Define environment variables ---
    export qafolder="${WORKSPACE}"
    export NUSOL_EXEC_DIR="${NUSOL_DLL_DIR}"

    export MS_USERDLL_DIR="${NUSOL_DLL_DIR}:${qafolder}/user_dll/tmc64:${qafolder}/user_dll/honda64"

    export SALT_LICENSE_SERVER="29000@blrpc891"
    export COSIN_LMSERVER="blrma-build-w10:80"
    export MADLIC_LICENSE_FILE="26000@blrma-qa-cent7"
    export ALTAIR_LICENSE_FOR_CDTIRE="1"
    export ALTAIR_LICENSE_PATH="6200@trlicsrv01.prog.altair.com"

    export RADFLEX_PATH="${qafolder}/license/radflex/linux64"
    export HW_MA_ALM_SUBSYS_ALT_IMAGE="${qafolder}/license/radflex/linux64/liblmx-altair.2025.0.0.so"

    # Commented out variables from original script remain commented
    #export ACTIVATE_ROOT="/opt/Altair_Installs/2025/TwinActivate2025"
    #export ACTIVATE_VERSION="2025"
    #export ACTIVATE_COSIM_USE_SMP="internal_use_only"

    export MS_PYTHON_DLL="nupy38"

    # IMPORTANT: Prepend to existing paths to ensure our directories are found first.
    # Also, avoid adding duplicate entries if the script is sourced multiple times.
    local new_ld_path="${NUSOL_DLL_DIR}:${RADFLEX_PATH}:${qafolder}/user_dll/tmc64:${qafolder}/user_dll/honda64:/home/msteam/python38/lib/"
    local new_path="${NUSOL_DLL_DIR}:${RADFLEX_PATH}:${qafolder}/user_dll/tmc64:${qafolder}/user_dll/honda64:/home/msteam/python38/bin/"

    # Safely prepend to PATH and LD_LIBRARY_PATH
    # This prevents adding the same paths over and over if you source the script again.
    if [[ ":$LD_LIBRARY_PATH:" != *":${new_ld_path}:"* ]]; then
        export LD_LIBRARY_PATH="${new_ld_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    if [[ ":$PATH:" != *":${new_path}:"* ]]; then
        export PATH="${new_path}${PATH:+:${PATH}}"
    fi

    # --- Step 3: Provide feedback to the user ---
    echo "? Environment variables have been set for workspace: ${WORKSPACE}"
    echo "   NUSOL_DLL_DIR is: ${NUSOL_DLL_DIR}"
    echo "   Run 'env | grep NUSOL' to see the new variables."

    # --- Step 4: Restore original shell options ---
    # This restores the shell to its state before the script was sourced.
    eval "$original_shell_options"
}

# Call the function with all the script's arguments
_setup_env "$@"
# Unset the function itself so it doesn't linger in the shell's environment
unset _setup_env
