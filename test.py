def find_longest_chain(list, condition):
    """find longest chain of elements in list that satisfy condition and return start and end index"""
    max_len = 0
    current_len = 0
    max_index = 0
    for i in range(len(list)):
        if condition(list[i]):
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_index = i - current_len
            current_len = 0

    if current_len > max_len:
        max_len = current_len
        max_index = i - current_len + 1

    return max_index, max_index + max_len


l = [2, 2, 1, 2, 3, 0, 4, 5, 6, 7]

s, e = find_longest_chain(l, lambda x: x != 0)
print(l[s:e])
