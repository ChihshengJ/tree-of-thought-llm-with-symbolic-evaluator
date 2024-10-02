import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import subprocess
import re

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    system_prompt = task.value_system_prompt_wrap() if task.system_prompt else ""
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, system_prompt=system_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def symbolic_evaluator(task, ys, cache_value=True) -> list[int]:
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:     
            current_outputs, note = task.get_temporary_output(y)
            # print("current_numbers:", current_outputs)
            if note != 1:
                value = note
            else:
                if cache_value and y in task.value_cache:
                    value = task.value_cache[y]
                else:
                    command = ['python3', task.evaluator] + list(map(str, current_outputs))
                    try:
                        result = subprocess.run(command, check=True, text=True, capture_output=True)
                        # print("result:", result.stdout.strip())
                        value = int(float(re.findall(r'\d+', result.stdout.strip())[-1]))
                        # print("value:", value)
                    except subprocess.CalledProcessError as e:
                        print("Error:", e)
                        value = 0
                    if cache_value:
                        task.value_cache[y] = value
            local_value_cache[y] = value
        values.append(value)
        # print("temp:", values)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    system_prompt = task.vote_system_prompt_wrap() if task.system_prompt else ""
    vote_outputs = gpt(vote_prompt, system_prompt=system_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    system_prompt = task.system_prompt_wrap() if task.system_prompt else ""
    proposals = gpt(propose_prompt, system_prompt=system_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    system_prompt = task.system_prompt_wrap() if task.system_prompt else ""
    samples = gpt(prompt, system_prompt=system_prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # early stop when an answer occurs
        early_stop = False
        # print("checking early stop")
        for y in ys:
            # print("y:", y)
            if task.is_answer_present(idx, y):
                early_stop = True
                break
        # print("checking complete")
        if early_stop:
            # print("early stop!!")
            infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': None, 'values': None, 'select_new_ys': None})
            break
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            # in this implementation the length of model output would be restricted to the argument truncate length to save time.
            new_ys = [
                (np.random.choice(proposals, size=args.truncate_length, replace=False).tolist() if len(proposals) > args.truncate_length else proposals)
                for y in ys
                for proposals in [get_proposals(task, x, y)]
            ]
        new_ys = list(itertools.chain(*new_ys))     


        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'symbolic':
            values = symbolic_evaluator(task, new_ys)
        
        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
            # print(select_ids)
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))

        # log
        if to_print:       
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
        if all(v == 0 for v in sorted_values):
            print("all zeros, exit")
            break        
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}