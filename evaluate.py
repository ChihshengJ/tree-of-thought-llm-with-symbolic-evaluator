import json
from pathlib import Path

def calculate_accuracy_from_log(file_path):
    if not Path(file_path).exists():
        print(f"No log file found at {file_path}")
        return
    
    # Read the log file
    with open(file_path, 'r') as f:
        logs = json.load(f)

    if not logs:
        print("Log file is empty.")
        return
    
    # Initialize counters
    cnt_avg = 0
    cnt_any = 0
    total_entries = len(logs)
    times = []

    for log in logs:
        infos = log.get('infos', [])
        if infos:
            accs = [info['r'] for info in infos]
            cnt_avg += sum(accs) / len(accs)  # Average accuracy per entry
            cnt_any += any(accs)  # Check if there's at least one correct answer
        time = log.get('time_spent')
        if time:
            times.append(log.get('time_spent'))

    average_accuracy = cnt_avg / total_entries
    any_accuracy = cnt_any / total_entries
    average_time = sum(times) / total_entries
    
    print(f"Total entries processed: {total_entries}")
    print(f"Average Accuracy: {average_accuracy:.2f}")
    print(f"Any Correct Accuracy: {any_accuracy:.2f}")
    print(f"Average time spent: {average_time:.2f}")

if __name__ == "__main__":
    log_file_path = input("log path:\n")
    calculate_accuracy_from_log(log_file_path)