import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import * 
from tot.models import gpt
from functools import partial


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4
        self.system_prompt = False
        self.evaluator = 'task_evaluator.py'

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def get_temporary_output(self, y: str) -> tuple[list[int], int]:
        lines = y.strip().split('\n') 
        # print("lines:", lines)   
        last_line = lines[-1]
        pattern = r"^\d+\s*[+\-*/]\s*\d+\s*=\s*-?\d+\s*\(left:\s*(?:-?\d+\s*)+\)$"
        # print("last_line:", last_line)
        if "Answer:" in last_line and "24" in last_line:
            lengths = {len(l.split("left: ")[-1].split(')')[0].split(" ")) for l in lines[:-1]}
            # print("possible candidate lengths", len(lengths))
            if len(lengths) < len(lines) - 2:
                return [], 0      
            expression = last_line.replace('Answer: ', '').split('=')[0]
            if sympy.simplify(expression) == 24:
                return [], 1000
        if re.match(pattern, last_line):
            lengths = {len(l.split("left: ")[-1].split(')')[0].split(" ")) for l in lines}
            # print("lengths", len(lengths))
            if len(lengths) < len(lines):
                return [], 0
            return [int(float(n)) for n in last_line.split('left: ')[-1].split(')')[0].split(" ")], 1
        else:
            return [], 0

    def set_system_prompt(self, activation: int):
        if activation == 1:
            self.system_prompt = True
        else:
            self.system_prompt = False      

    def set_evaluator(self, model: str):
        if self.evaluator and os.path.exists(self.evaluator) and os.path.getsize(self.evaluator) > 0:
            print("Using existing evaluator.")
            pass
        else:
            gpt_eva = partial(gpt, model=model, temperature=1)
            print(gpt_eva)
            try:
                evaluator_script = gpt_eva(evaluator_prompt, model=model, n=1, stop=None)[0]
                assert isinstance(evaluator_script, str)
                file_path = 'task_evaluator.py'
                with open(file_path, 'w') as file:
                    lines = evaluator_script.split('\n')
                    # print("lines:", lines)
                    start = next(i for i, line in enumerate(lines) if "import" in line)
                    end = next(i for i, line in enumerate(lines) if line.strip() == '```' and i > start)
                    code = '\n'.join(lines[start-1 + 1:end])
                    # print("code:", code)
                    file.write(code)
                self.evaluator = file_path
                print("Task evaluator generated.")
            except Exception as e:
                print("An error occurred when setting evaluator:", e)
        
    def is_answer_present(self, idx:int, y:str) -> int:
        if "Answer:" in y:
            print("possible:", y)
            # Answer: (8 / 2) * (4 * 8) = 24\n
            problem_numbers = sorted(re.findall(r'\d+', self.data[idx]))
            last_line = y.strip().split('\n')[-1]
            expression = last_line.replace('Answer: ', '').split('=')[0]
            numbers = sorted(re.findall(r'\d+', expression))
            # print("problem:", problem_numbers)
            # print("numbers:", numbers)
            if '24' in last_line and numbers == problem_numbers and sympy.simplify(expression) == 24:
                    return True
            else:
                return False
        else:
            return False

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        # print("test expression:", expression)
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
        

    @staticmethod
    def system_prompt_wrap() -> str:
        return system_prompt
    
    @staticmethod
    def value_system_prompt_wrap() -> str:
        return value_system_prompt
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = last_step_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt 
        
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        """
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value_map[keyword] for output in value_outputs 
                                        for line in output.split('\n') 
                                        for keyword in value_map if keyword in line)
        """
        # print("value:", value)
        return value
    