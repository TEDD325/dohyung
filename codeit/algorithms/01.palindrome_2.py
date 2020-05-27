import math

def is_palindrome(word):
    floor_length_word = math.floor(len(word) / 2)
    result = False
    result_sum = 0

    for idx in range(floor_length_word):
        if word[idx] == word[-1 - idx]:
            result_sum += 1

    if result_sum == floor_length_word:
        result = True

    return result


# 테스트
print(is_palindrome("racecar"))
print(is_palindrome("stars"))
print(is_palindrome("토마토"))
print(is_palindrome("kayak"))
print(is_palindrome("hello"))