# 숫자야구게임 구현
import random

number_list = list(range(0, 10))  # 0~9 list 생성
answer_list = list()  # 정답 list 생성

# 게임할 digit을 사용자가 입력
while True:  # 무한 loop
    try:
        input_digit = int(input("자릿수를 입력하세요."))  # digit(bool->int)
        if input_digit <= 0:  # 음수나 0 입력한 경우 예외처리
            print("1 이상의 수를 입력하세요.")
            continue
    except ValueError:  # 숫자 아닌 걸 입력한 경우 예외처리
        print("숫자만 입력할 수 있습니다.")
        continue
    answer_list = random.sample(number_list, input_digit)  # 간단한 version
    print(f"{input_digit}자리의 숫자야구를 시작합니다.")
    break  # loop 종료
    # while len(answer_list) < input_digit: #고안했던 version
    # zerotonine_int = int(random.random() * 10)  # 0~9 int
    # if number_list[zerotonine_int] not in answer_list:
    # answer_list.append(number_list[zerotonine_int])  # 0~9 list에서 무작위 index를 골라 정답 list에 넣음
    # 연산 횟수가 많아질 수 있다는 문제가 있음(pop을 활용하면 len이 달라져서 index 참조 불가능)

trial_count = 0
result_bool = False

# 답안 입력과 결과 판정
while result_bool is False:  # result_bool이 True가 될 때까지 loop
    trial_count += 1  # 시도 횟수(int)
    trial = input(f"{trial_count}번째 시도입니다. 답안을 입력하세요.")  # 답안(str)
    if len(trial) != input_digit:  # digit을 다르게 입력했을 때
        print(f"{input_digit}자리의 답안이 아닙니다.")
        continue
    try:
        int(trial)
    except ValueError:  # 숫자 아닌 걸 입력한 경우 예외처리
        print("숫자만 입력할 수 있습니다.")
        continue

    trial_list = [int(digit) for digit in trial]  # 답안(str->int->list)
    overlap_bool = False

    # 중복 답안 검사
    for digit in range(len(trial_list)):
        overlap_count = trial_list.count(trial_list[digit])
        if overlap_count > 1:
            overlap_bool = True
            break
    if overlap_bool is True:
        print("중복된 숫자는 입력할 수 없습니다..")
        continue

    #답안을 맞힌 갯수 검사
    strike_count = 0
    ball_count = 0

    for digit in range(len(answer_list)):
        if trial_list[digit] == answer_list[digit]:  # 수, 자리가 맞았을 때
            strike_count += 1
        elif trial_list[digit] in answer_list:  # 수만 맞았을 때
            ball_count += 1

    if strike_count == 3:
        print(f"{trial_count}번째 시도에서 정답 {answer_list}을 맞혔습니다. YOU WIN!")
        result_bool = True  # loop 종료
    elif strike_count != 0 or ball_count != 0:
        print(f"S: {strike_count} B: {ball_count}")
    else:
        print("OUT")

# strike, ball count 부분을 모듈화할 수 있지 않을까?
