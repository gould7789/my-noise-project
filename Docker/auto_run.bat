@echo off
SET project_name=noise-analyzer

echo [1] 🔧 Docker image build...
docker build -t %project_name% .

echo [2] 🧭 Open VSCode folder...
start .

echo [3] 🐳 Stop previous container if exists...
docker rm -f %project_name% >nul 2>&1

echo [4] 🚀 Run Docker container...
docker run -it -v "C:\Users\user\my-noise-project:/app" --rm --name %project_name% %project_name%

echo.
echo ✅ Docker container closed. Type 'exit' to quit.
cmd /k