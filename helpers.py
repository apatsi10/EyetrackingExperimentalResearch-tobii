import os

def average_list(list):
    sum_list = 0
    for i in range(len(list)):
        sum_list = sum_list + list[i]
    return sum_list / len(list)

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

