from run_mujoco import train
import submitit

executor = submitit.AutoExecutor(folder="log_test")
executor.update_parameters(timeout_min=1, partition="scavenge")
job = executor.submit(train, "Pendulum-v0", 10, 0, 1.0, 1, True)

output = job.result()
