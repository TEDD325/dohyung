"""
우선 랜덤이 무엇인지부터 살펴볼까요.
주사위를 던지는 상황을 생각해봅시다. 주사위의 각 면에는 1개에서 6개까지의 눈이 새겨져 있어서, 주사위를 던질 때마다 그 중 하나의 숫자가 선택됩니다.
주사위를 직접 던져보기 전에는 다음번에 어떤 숫자가 나올지 알 수가 없죠. --> 각 사건이 independent하다는 의미
그런데 주사위를 600번 정도 던져보면 각 숫자가 대략 100번 정도는 나오기는 합니다. --> 정규분포를 따른다는 의미
이런 것이 바로 난수(random number)입니다.
"""

"""
그 중 아무 원소나 하나 뽑아주는 함수가 choice() 함수
"""
import random as pr

abc = ['a', 'b', 'c', 'd', 'e']
pr.shuffle(abc) # shuffle() 함수는 순서형 자료(sequence)를 뒤죽박죽으로 섞어놓는 함수. return이 None이므로, 객체 자체를 바꾼다.
# abc = pr.shuffle(['a', 'b', 'c', 'd', 'e'])
random_val = pr.choice(abc)
print(random_val)