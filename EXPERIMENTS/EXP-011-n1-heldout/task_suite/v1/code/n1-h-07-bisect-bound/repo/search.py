"""Binary search helpers."""


def find_last_at_most(values, target):
    low, high = 0, len(values) - 1
    answer = -1
    while low < high:
        mid = (low + high) // 2
        if values[mid] <= target:
            answer = mid
            low = mid + 1
        else:
            high = mid - 1
    return answer
