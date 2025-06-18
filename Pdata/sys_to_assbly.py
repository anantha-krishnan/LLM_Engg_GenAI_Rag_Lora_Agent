import re
import os
from pathlib import Path
from ParseModelSystem import ModelSys
from AssemblyFileCreation import AssemblyFileCreation
from utility_fncs import UtilityFunctions

class SysToAsm:
    """
    Class to convert system definitions to assembly files.
    """

    def __init__(self):
        """
        Initialize the SysToAsm class with utility, assembly, and model system helpers.
        """
        self.utfncs = UtilityFunctions()
        self.asm_funcs = AssemblyFileCreation()
        self.model_sys = ModelSys()

    def __del__(self):
        """
        Destructor for cleanup if needed.
        """
        pass

