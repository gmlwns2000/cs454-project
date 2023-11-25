import os

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

def add_str(op1, op2):
    op1 = int(op1)
    op2 = int(op2)
    result = op1 + op2
    result_str = str(result)
    return result_str


if __name__ == '__main__':
    in1 = '14'
    in2 = '11'
    add = add_str(in1, in2)
    print(add)
