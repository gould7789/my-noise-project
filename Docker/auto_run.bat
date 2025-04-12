echo off
SET project_name=noise-analyzer

echo [1] Docker image build...
docker build -t %project_name% .

echo [2] VSCode opne
start .

echo [3] Docker running a container
@REM docker run -it --rm -v %cd%:/app --name %project_name% %project_name%
docker run -it -v "C:\Users\user\my-noise-project:/app" --rm --name %project_name% %project_name%