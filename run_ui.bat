@echo off
setlocal

set "ROOT=%~dp0"

pushd "%ROOT%"

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    popd
    echo Python 3.11 was not found. Install Python 3.11 to launch the UI.
    exit /b 1
)

py -3.11 -m zlsde.ui_simple
set "EXIT_CODE=%errorlevel%"
popd
exit /b %EXIT_CODE%
