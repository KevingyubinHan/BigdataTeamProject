#

def add_mul(choice, *args):
    if choice == 'add':
        result = 0
        for i in args:
            result += i
    elif choice == 'mul':
        result = 1
        for i in args:
            result *= i
    return result


def print_kwargs(**kwargs):
    print(kwargs)


add = lambda a, b: a + b
result = add(3, 4)


#

def is_odd(a):
    if a % 2 == 0:
        print('짝수입니다.')
    elif a % 2 != 1:
        print('홀수입니다.')


is_odd = lambda x: True if x % 2 == 1 else False


def avarge_all(*args):
    sum_args = 0
    for i in args:
        sum_args += i
    return sum_args / len(args)


input1 = int(input("첫번째 숫자를 입력하세요:"))
input2 = int(input("두번째 숫자를 입력하세요:"))

total = input1 + input2
print("두 수의 합은 %s 입니다" % total)

#

korean, english, mathematics, science = 100, 86, 81, 91


def max_score(*args):
    m = max(args)
    return m


def is_palindrome(word):
    if len(word) < 2:
        return True
    if word[0] != word[-1]:
        return False
    return is_palindrome(word[1:-1])


files = ['font', '1.png', '10.jpg', '11.gif', '2.jpg', '3.png', 'table.xslx', 'spec.docx']

print(list((filter(lambda x: x.find('.jpg') != -1 or x.find('.png') != -1, files))))


# class, method, instance, attribute

class Character:  # 클래스 첫글자는 대문자로 작성
    def __init__(self, name, nickname):  # __init__:생성자. 만들어지자마자 자동으로 호출되도록 한 함수.
        self.name = name  # 속성
        self.nickname = nickname

    def greeting(self):  # 클래스 내 메서드. self를 항상 받고 추가 인자도 받을 수 있음
        print(f"저는 {self.name}입니다. {self.nickname} 배역입니다.")


bucky = Character('bucky', 'winter soldier')  # bucky는 Character의 인스턴스
bucky.greeting()  # self는 앞의 변수로 이미 배정되었음
steve = Character('steve', 'captain america')
steve.greeting()
print(bucky.name, bucky.nickname)  # 클래스로 만든 객체의 객체변수는 다른 객체의 객체변수에 상관없이 독립적인 값을 유지한다.


class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, val):
        self.value += val


class UpgradeCalculator(Calculator):  # 클래스 상속. 다른 모든 특징을 유지
    def minus(self, val):
        self.value -= val


cal = UpgradeCalculator()
cal.add(10)
cal.minus(7)
print(cal.value)


class MaxLimitCalculator(Calculator):
    def add(self, val):
        self.value += val
        if self.value > 100:
            self.value = 100


cal = MaxLimitCalculator()
cal.add(50)
cal.add(60)
print(cal.value)

#메뉴 Class

import pandas as pd

filepath = './menu_excel.xlsx'
menu_excel = pd.read_excel(filepath)
menu_excel

class Menu:
    def __init__(self):
        self.menu1 = menu_excel['메뉴명'][0]
        self.menu1_cost = menu_excel['가격'][0]
        self.menu1_quantity = 0

    def quantity_changed(self, num):
        if self.menu1_quantity + num < 0:
            self.menu1_quantity = 0
        else:
            self.menu1_quantity += num
        return self.menu1_quantity

    def total_amount(self):
        self.tot_amount = self.menu1_quantity * self.menu1_cost
        return self.tot_amount

americano = Menu()
americano.quantity_changed(1)
americano.quantity_changed(-1)
americano.total_amount()

class Knight():
    def __init__(self, health, mana, armor):
        self.health = health
        self.mana = mana
        self.armor = armor

    def slash(self):
        print('베기')


x = Knight(health=542.4, mana=210.3, armor=38)
print(x.health, x.mana, x.armor)
x.slash()

class temp_cls:
    tmp = 0
    def __init__(self):
        temp_cls.tmp += 1

    @classmethod
    def print_tmp(cls):
        print(f'{cls.tmp}입니다.')

    @classmethod
    def create_temp(cls):
        instance = cls()
        return instance

temp = temp_cls()
temp_cls.print_tmp()
temp_cls.create_temp()

class Firstclass:
    def __init__(self, num):
        self.num = num

    def numplus(self):
        self.num += 1
        return self.num

inst = Firstclass(0)

class Secondclass(Firstclass):
    @classmethod
    def create(cls):
        inst_create = cls(0)
        return inst_create



class Testclass:

    clsatt = 0

    def __init__(self):
        self.attribute = 0

    def testFunc1(self):
        self.attribute = 1

        self.testFunc2()

        return self.attribute

    def testFunc2(self):
        Testclass.clsatt = 10
        print('func2')
        return Testclass.clsatt

test = Testclass()
test.testFunc1()

temp_list = [0,1,2,3]

for i in range(4):
    temp_list[i] = Test(i)

testinst = Test()
temp_list = [testinst]



class Date():
    def is_date_valid(self):
        year, month, day = map(int, self.split('-'))
        return month <= 12 and day <= 31


if Date.is_date_valid('2000-13-31'):  # 클래스 호출
    print('올바른 날짜 형식입니다.')
else:
    print('잘못된 날짜 형식입니다.')

x = AdvancedList([1, 2, 3, 1, 2, 3, 1, 2, 3])
x.replace(1, 100)
print(x)


class AdvancedList(list):
    def replace(self, a, b):
        while self.count(a) != 0:
            self[self.index(a)] = b


# 리스트 응용
temp_list = ['first avenger', 'winter soldier', 'civil war']
temp_list.append('infinity war')  # 리스트의 끝에 새로운 요소 하나를 추가
temp_list.extend(['infinity war', 'endgame'])  # 리스트의 끝에 여러 요소를 추가
temp_list.insert(2, 'insertion')  # 인덱스에 있도록 요소를 추가
temp_list[2:2] = ['insertion', 'insertion']  # 인덱스에 있도록 여러 요소를 추가(슬라이싱)
temp_list
temp_list.remove('insertion')  # 특정 값을 찾아 삭제. 가장 처음 있는 값만 삭제됨
temp_list.index('winter soldier')  # 특정 값의 인덱스를 반환. 가장 처음 있는 값으로.
temp_list.count('first avenger')  # 특정 값의 갯수를 반환
for i in temp_list:
    print(i)
for index, value in enumerate(temp_list):
    print(index, value)

# 리스트 내포
temp_list = [i for i in range(10) if i % 2 == 0]

# map
temp_list = [1.2, 2.3, 3.5]
for i in range(len(temp_list)):  # 정수화시킬 때 for를 이용하면
    temp_list[i] = int(temp_list[i])
temp_list = list(map(int, temp_list))  # iterable의 요소를 함수로 처리해줌(새로운 리스트 반환)
map_list = list(map(str, range(3)))
m = map(int, input().split())
a, b = m

# 연습문제
a = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india']
answer = [i for i in a if len(i) == 5]
answer

# 문자열
' '.join(temp_list)  # 리스트+구분자 join해서 str으로 반환
'11'.zfill(4)  # 0+문자열이 입력한 길이가 되도록 앞에 0을 채움
'first avenger'.find('av')  # 찾는 문자열의 인덱스 반환. 없으면 -1 반환. 가장 처음 있는 값으로
'winter soldier'.index('sol')  # find와 같지만 없을 시 에러

# 디렉토리 추가
import sys

sys.path.append('C:\\Users\\JH')

# 모듈 불러오기
import module_test

bucky = module_test.Character('bucky', 'winter soldier')  # 클래스 앞에 module 명시
bucky.greeting()
bucky.ishebucky()
module_test.tilltheendoftheline()  # 함수 앞에 module 명시


# filter 실습
def positive(x):
    return x > 0


question = [0, -3, -5, 1, 2, 6, 9]
list(filter(positive, question))  # filter: 함수, 변수를 입력받아 참인 값만 반환
list(filter(lambda x: x > 0, question))  # 같은 반환이지만 lambda로 간단하게


def odd(x):
    if x % 2 != 0:
        return x


list(filter(odd, question))
list(filter(lambda x: x % 2 != 0, question))

# 내장함수
isinstance(bucky, module_test.Character)  # isinstance: 객체, 클래스를 입력받아 그 클래스의 인스턴스이면 참 반환
list(map(positive, question))  # map: 함수, 반복가능 객체를 입력받아 결괏값 반환
list(map(lambda x: x * 2, question))  # map과 lambda. 작은 for문으로 생각?
ord('1')  # ord: 문자열을 입력받아 ASCII 코드 값을 반환 <->chr(c)
list(range(1, 10, 2))  # range: 시작값(default 0),끝날값(-1),step을 입력받아 반복가능객체로 반환
round(3.141592, 2)  # round: 수와 ndigits을 입력받아 ndigits까지 반올림한 수를 반환
sorted(question)  # sorted: 반복가능객체를 입력받아 정렬한 후 리스트로 반환
list(zip(['a', 'b', 'c'], [1, 2, 3]))  # zip: 반복가능객체를 입력받아 동일한 len이면 묶어서 반환

# 연습문제
temp_list = [1, -2, 3, -5, 8, -3]
list(filter(lambda x: x > 0, temp_list))
temp_list = [1, 2, 3, 4]
list(map(lambda x: x * 3, temp_list))
round(17 / 3, 4)

# 로또 만들기
import random


def choice_pop(data):
    num = random.choice(data)
    data.remove(num)
    return num


def lotto_pop():
    lotto_list = []
    num_list = list(range(1, 46))
    while len(lotto_list) < 6:
        lotto_list.append(choice_pop(num_list))
    return sorted(lotto_list)


def n_lotto(n):
    n_lotto_list = []
    while len(n_lotto_list) < n:
        n_lotto_list.append(lotto_pop())
    return n_lotto_list


# 예외처리
try:
    file = open('maria.txt', 'r')
except FileNotFoundError:
    print('file not found!')
else:
    s = file.read()
    file.close()

#
from mymath import arithmetic

arithmetic.add(1, 3)

import mymath

mymath.add(1, 3)
mymath.mean([1, 2, 3, 4, 5])

from mymath import pi

pi

#
joinstr_list = ['^(N_)?(\(주\)|주식회사|\(?주\)한무쇼핑|주\)|한무쇼핑\(주\))?(.*)(매장|-)', \
                '^(N_)?(\(주\)|주식회사|\(?주\)한무쇼핑|주\)|한무쇼핑\(주\))?(.*)']

id_list = ['로라메르시에', '엘리자베스아덴', '르라보', '킬리안', '하이코스', '비디비치', \
           '숨', '산타마리아노벨라', '오리진스', '라메르', '달팡', '그라운드플랜', '데코르테', \
           '동인비', '톰포드뷰티', '에르메스퍼퓸', '라페르바', '불리1803', '끌레드뽀보떼', 'RMK', '구딸파리', \
           '시슬리화장품', '지방시뷰티', '라프레리']

complete_str_list = []


def join_str_id(id_list, str_list, result_list):
    for i in id_list:
        for j in str_list:
            result_list.append(j + i)


join_str_id(id_list, joinstr_list, complete_str_list)

for i in complete_str_list:
    print(i)


# 게시판 페이징
def getTotalPage(tot_num, onepage_tot):
    if tot_num % onepage_tot == 0:
        return tot_num // onepage_tot
    else:
        return tot_num // onepage_tot + 1


# web scraping
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup

daum_url = 'https://media.daum.net/'
html = urlopen(daum_url)
bsObj = BeautifulSoup(html, "html.parser")
main_news = bsObj.find('ul', {'class' : 'list_headline'}) #뉴스 헤드라인
a_all = main_news.find('li').find_all('a')
print(a_all[0].get_text().strip())

naver_url = 'https://www.naver.com/'
resp = requests.get(naver_url)
soup = BeautifulSoup(resp.text, 'html.parser')
titles = soup.select('.ah_roll .ah_k')
for i, title in enumerate(titles, 1):
    print(f'{i}위 {title.get_text()}')

