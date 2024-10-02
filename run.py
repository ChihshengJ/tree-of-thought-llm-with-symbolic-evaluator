import os
import json
import argparse
from tqdm import tqdm
import time

from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve
from tot.models import gpt_usage

def run(args):
    task = get_task(args.task)
    task.set_system_prompt(args.system_prompt)
    task.set_evaluator(model='gpt-4')
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f'./new_logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_system_prompt{args.system_prompt}.json'
    else:
        file = f'./new_logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_truncate{args.truncate_length}_start{args.task_start_index}_end{args.task_end_index}_system_prompt{args.system_prompt}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    if os.path.exists(file):
        with open(file, 'r') as f:
            previous_logs = json.load(f)
            if previous_logs:
                last_processed_index = previous_logs[-1]['idx'] + 1
                args.task_start_index = max(args.task_start_index, last_processed_index)
                logs = previous_logs  # Reuse existing logs
                
    for i in tqdm(range(args.task_start_index, args.task_end_index), desc="Processing"):
        start_time = time.time()
        ys, info = (naive_solve if args.naive_run else solve)(args, task, i)
        elapsed_time = time.time() - start_time
        infos = [task.test_output(i, y) for y in ys]
        info.update({
            'idx': i,
            'ys': ys,
            'infos': infos,
            'usage_so_far': gpt_usage(args.backend),
            'time_spent': elapsed_time
        })
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')

    n = max(1, args.task_end_index - args.task_start_index)
    print(f"Average Accuracy: {cnt_avg / n}, Any Accuracy: {cnt_any / n}")
    print('Usage so far:', gpt_usage(args.backend))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4o', 'gpt-3.5-turbo', 'gpt-4o-mini', 'llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'claude-3-5-sonnet-20240620', "claude-3-sonnet-20240229", "gemini-1.5-flash", "gemini-1.5-pro"], default='gpt-3.5-turbo')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote', 'symbolic'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--truncate_length', type=int, default=10000)
    args.add_argument('--system_prompt', type=int, default=0)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)