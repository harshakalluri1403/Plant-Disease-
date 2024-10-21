import subprocess
import time

# Define the list of epochs
epochs_list = [100, 200, 500, 1000]

def run_training(epochs):
    print(f"Starting training with {epochs} epochs...")
    # Open the log file in write mode
    log_file = open(f"output_{epochs}.log", "w")
    # Use subprocess.Popen to run the command and redirect output to the log file
    process = subprocess.Popen(["python", "train_model.py", "--epochs", str(epochs)],
                               stdout=log_file,
                               stderr=subprocess.STDOUT)
    return process

def main():
    processes = []
    for epochs in epochs_list:
        process = run_training(epochs)
        processes.append(process)
        time.sleep(5)  # Optional delay to ensure the previous command starts

    # Wait for all processes to complete
    for process in processes:
        process.wait()

if __name__ == "__main__":
    main()

