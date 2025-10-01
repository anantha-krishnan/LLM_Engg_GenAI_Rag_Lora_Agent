:: This simple bat file sets up the environment for MS
:: It takes the x64 or x32 input argument and sets us 
:: the environment variables accordingly. 
@echo off


rem echo "input argument are %1 , %2 , %3"
echo Setting DLL...
rem TO SET
set ALTAIR_INSTALLATION=E:\install\2026_0_16\hwsolvers\motionsolve\bin\win64
rem TO SET
set ALTAIR_HOME=E:\install\2026_0_16
rem 2020.0.0.50-raghpMauto_20BFix_BW-dev55
echo "Setting up for %ALTAIR_INSTALLATION%"
set ALTAIR_LICENSE_PATH=6200@trlicsrv01.prog.altair.com
set NUSOL_DLL_DIR=%ALTAIR_INSTALLATION%
rem D:\Workspaces\Solver\anantk_BLRLAP1017_releaseBinaries

set NUSOL_EXEC_DIR=%NUSOL_DLL_DIR%
rem set MS_USERSUBDLL=%NUSOL_DLL_DIR%\msauto\msautoutils.dll
set NUSOLQA_HWD_DIR=%ALTAIR_HOME%\hwdesktop\hw\bin\win64
set PYTHONHOME=%ALTAIR_HOME%\common\python\python3.10\win64
set SALT_LICENSE_SERVER=29000@blrpc891
title Workspace Build
echo Setting Radflex path...
set RADFLEX_PATH=%ALTAIR_HOME%\hwsolvers\common\bin\win64
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