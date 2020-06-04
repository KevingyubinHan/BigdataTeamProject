# 숫자야구게임 구현
import random

# while문으로 try를 감싸서 보내기
# 굳이 def로 정의할 필요 없고 모듈화
# choices 말고 random.random 써보기
# 문자열 포매팅에 익숙해지고 f포매팅 해보기
# 리스트내포에는 []
# for 안에서 else를 쓰지 않아도 돌아감
# 인터페이스를 사용자친화적으로. split보다 다른 걸 생각해보자
# ctrl alt l 하면 indent 알아서 정렬

while True:
    try:
        input_digit = int(input("자릿수를 입력하세요."))  # digit 사용자입력 및 int 캐스팅
    if input_digit <= 0:  # 음수나 0 입력한 경우 예외처리
        print("1 이상의 수를 입력하세요.")
        continue
    except ValueError:  # 숫자 아닌 걸 입력한 경우 예외처리
        print("숫자만 입력할 수 있습니다.")
        continue


number_list = list(range(0, 10))  # 0~9 list 생성
answer_list = list()  # 정답(int) list 생성

while len(answer_list) < input_digit: # 사용자입력 digit과 정답의 digit이 같아질 때까지 loop
    answer_list = random.choices(number_list, k=input_digit)  # 사용자입력 digit만큼 추출(중복가능)
    answer_set = set(answer_list)  # set 캐스팅으로 중복 제거
    answer_list = list(answer_set)  # 다시 list 캐스팅해서 len 판정과 index 생성
print(f"{input_digit}자리의 숫자야구를 시작합니다.")  # f문자열 formatting

answer_str_list = list(str(args) for args in answer_list)  # 정답(str) list 생성
answer_str_list = [str(digit) for digit in answer_list]
answer_count = 0  # 시도 횟수 int 생성

while True:  # 무한 loop
    answer_count += 1  # 시도 횟수 증가
    print(f"{answer_count}번째 시도입니다.")
    gamer_answer = input("답안을 한 칸 씩 띄어쓰세요.").split()  # 공백울 기준으로 list 캐스팅
    if gamer_answer == answer_str_list:  # 입력이 정답과 같은 경우
        print(f"승리하셨습니다. 시도한 횟수: {answer_count} 게임을 종료합니다.")
        break  # 게이머가 승리한 경우 loop 종료
    else:  # 입력이 정답과 다른 경우
        strike_count = 0
        ball_count = 0
        for index in range(len(answer_str_list)):
            if gamer_answer[index] == answer_str_list[index]:
                strike_count += 1  # 수, 자리까지 모두 같은 경우 strike
            elif gamer_answer[index] in answer_str_list:
                ball_count += 1  # 수만 같은 경우 ball
            else:
                continue  # 모두 틀린 경우 다음 자리로 continue
        if strike_count > 0 or ball_count > 0:
            print(f"스트라이크: {strike_count} 볼: {ball_count}")
        else:
            print("아웃입니다.")
