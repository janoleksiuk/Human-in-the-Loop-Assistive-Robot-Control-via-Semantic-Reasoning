@echo off
REM Legacy Windows helper for preprocessing one activity class.

set /p activity="Enter activity name (e.g. drinking, eating, running): "

python data_preprocessing.py ^
    --input_dir ..\..\..\data\har_dataset\raw\%activity% ^
    --output_path ..\..\..\data\har_dataset\processed\%activity%.npz ^
    --class_name %activity% ^
    --smooth sg ^
    --seq_len 30

pause
