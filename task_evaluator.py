from itertools import permutations, product

def solve(nums):
    ops = ['+', '-', '*', '/']

    for num_permu in permutations(nums):
        for ops_permu in product(ops, repeat=len(nums)-1):
            expr = ''
            for n, op in zip(num_permu, ops_permu):
                expr += f'{n}{op}'
            expr += str(num_permu[-1])

            try: 
                if abs(eval(expr) - 24) < 1e-6:
                    return 100
            except ZeroDivisionError:
                continue

    # find the range of results
    for num_permu in permutations(nums):
        for ops_permu in product(ops, repeat=len(nums)-1):
            expr = ''
            for n, op in zip(num_permu, ops_permu):
                expr += f'{n}{op}'
            expr += str(num_permu[-1])

            try: 
                result = eval(expr) 
                if 10 <= abs(result) <= 40:
                    return 50
            except ZeroDivisionError:
                continue

    return 0

def main():
    import sys
    nums_list = list(map(int, sys.argv[1:]))
    print(solve(nums_list))

if __name__ == '__main__':
    main()
