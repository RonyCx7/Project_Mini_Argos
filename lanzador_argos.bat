@echo off
title Project mini-Argos v1 - Launcher
color 0A

:: Establecer el directorio de trabajo para evitar problemas de rutas relativas
cd /d "C:\dev\Project_Mini_Argos_v1"

echo =======================================================
echo.
echo      [P R O J E C T   M I N I - A R G O S   v 1]
echo.
echo =======================================================
echo Iniciando el ecosistema de seguridad...
echo.

echo [1/3] Lanzando Servidor RTMP (MediaMTX) en segundo plano...
start "MediaMTX Server" "C:\dev\Project_Mini_Argos_v1\mediamtx_v1.16.3_windows_amd64\mediamtx.exe"

echo [2/3] Esperando 3 segundos para estabilizar el servidor...
timeout /t 3 /nobreak >nul

echo [3/3] Ejecutando interfaz principal de IA (PyQt6 + YOLOv8 + FaceNet)...
"C:\dev\Project_Mini_Argos_v1\env\Scripts\python.exe" "C:\dev\Project_Mini_Argos_v1\app.py"

echo.
echo Aplicacion finalizada. Cerrando...
exit
