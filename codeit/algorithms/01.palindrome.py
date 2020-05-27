def is_palindrome(word):
    length_word = len(word)
    temp_word = ""
    for i in range(1, length_word+1):
        temp_word += word[length_word-(i)]

    # print("[DEBUGGING POINT] - temp_str: ", temp_str)
    if (word == temp_word):
        return True
    else:
        return False
    # return

print(is_palindrome("racecar"))
print(is_palindrome("stars"))
print(is_palindrome("토마토"))
print(is_palindrome("kayak"))
print(is_palindrome("hello"))