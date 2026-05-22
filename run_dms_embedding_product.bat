@echo off
setlocal

@REM cd /d F:\baoyantest\dms

python scripts\run_task_loop_batch.py --task AudioRecorderRecordAudio --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task AudioRecorderRecordAudioWithFileName --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task ClockTimerEntry --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task ContactsAddContact --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task ExpenseAddSingle --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task FilesMoveFile --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task MarkorCreateFolder --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task MarkorEditNote --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task SimpleSmsSend --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task SystemWifiTurnOff --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch
python scripts\run_task_loop_batch.py --task TurnOnWifiAndOpenApp --runs 5 --memory_backend dms -- --dms_retrieval_mode embedding_product --skip_emulator_launch

echo.
echo DMS embedding_product batch finished.
pause