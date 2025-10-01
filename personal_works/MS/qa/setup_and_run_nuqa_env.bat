:: This simple bat file sets up the environment for MS
:: It takes the x64 or x32 input argument and sets us 
:: the environment variables accordingly. 
@echo off


rem echo "input argument are %1 , %2 , %3"
echo Setting DLL...
set ALTAIR_INSTALLATION=C:\Altairwin64\11.0\hwsolvers\bin\win64
set ALTAIR_HOME=C:\Altair_Installs\2024.0.0.2
rem 2020.0.0.50-raghpMauto_20BFix_BW-dev55
echo "Setting up for %ALTAIR_INSTALLATION%"

set NUSOL_DLL_DIR=%ALTAIR_INSTALLATION%
rem D:\Workspaces\Solver\anantk_BLRLAP1017_releaseBinaries

set NUSOL_EXEC_DIR=%NUSOL_DLL_DIR%
rem set MS_USERSUBDLL=%NUSOL_DLL_DIR%\msauto\msautoutils.dll
set NUSOLQA_HWD_DIR=%ALTAIR_HOME%\hwdesktop\hw\bin\win64
set PYTHONHOME=%ALTAIR_HOME%\common\python\python3.8\win64
set SALT_LICENSE_SERVER=29000@blrpc891
title Workspace Build
echo Setting Radflex path...
set RADFLEX_PATH=D:\Workspaces\QA\GitLab\qa\license\radflex\win64
set PATH=%PYTHONHOME%;%NUSOL_DLL_DIR%;%path%
rem ;C:\WINDOWS;C:\Program Files (x86)\Intel\Compiler\Fortran\10.0.026\IA32\Lib
set HW_MA_ALM_SUBSYS_ALT_IMAGE=%RADFLEX_PATH%\liblmx-altair.2025.0.0.dll

rem %PYTHONHOME%\python.exe scripts\NuQaReport.py qa_cmd\qa_model_run.xml
rem change to the working directory passed as the 3rd argument
rem echo Changing to %2% ...
chdir /d  %2
rem echo "Current working directory:"
rem cd

nuqa qa_cmd\qa_depot_xml_tire.xml +491 -%1