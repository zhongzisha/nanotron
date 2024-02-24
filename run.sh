CONFIG_FILE=${1}
export CUDA_DEVICE_MAX_CONNECTIONS=1
for jobid in `ps -u | grep python | awk '{ print $2 }'`; do kill $jobid; done
sleep 1
torchrun --nproc_per_node=8 run_train.py --config-file ${CONFIG_FILE} > log.txt 2>&1 &