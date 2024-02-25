export CUDA_DEVICE_MAX_CONNECTIONS=1
for jobid in `ps -u | grep python | awk '{ print $2 }'`; do kill $jobid; done
sleep 1
python examples/bench_llama_7b_8v100s_2048.py > log.txt 2>&1 &
