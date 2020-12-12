## 주석의 방법1
'''
변수(variable)
- 형식) 변수명 = 값 or 수식, 변수명
- 변수는 자료(객체)가 저장된 메모리이름이다.
- type은 없다. : R과 동일하다. ex) c언어 -> int, num, char

'''

# 1. 변수와 자료(data type)
var1 = 'Hello Python'
var2 = "Hello Python"
print(var1,var2)
print(type(var1)) # <class 'str'>
print(type(var2)) # <class 'str'> - 문자

var1 = 100
print(var1);print(type(var1)) #100 <class 'ínt'>  - 정수형
# 빌트인 펑션이라고 한다. 기본 내장함수


var3 =150.245
print(var3,type(var3)) #150.245 <class 'float'> - 실수형


var4 = True # 논리형 상수(True, False)
print(type(var4)) #<class 'bool'> - 논리형


# 2. 변수명 작성 규칙 (ppt.12)
'''
- 첫자 : 영문자 or _ (숫자,특수문자는 사용불가)
- 대소문자 구분: Score, score은 서로 다른 변수로 인식된다.
- 낙타체 : 두 단어 이상 결합시 사용(korScore) 두번째단어의 첫글짜를 대문자로 해준다.
- 키워드(명령어) 또는 함수명, 한글은 비권장 - 유니코드 때문에 오류가 날수있다.
'''

_num10 = 10 # 첫자만 숫자가 불가능하고 뒤에는 숫자가 가능하다.
_Num10 = 20 # num10과 다른 변수이다.

print(_num10*2) # 20
print(_Num10*2) # 40

# 명령어 확인하는 방법
# 특정 라이브러리를 가져와야된다. import는 외부 라이브러리를 가져올때 사용된다.
import keyword # keyword라는 모듈을 임포트한다.

py_keyword = keyword.kwlist # 모듈.호출가능한멤버 .
print("파이썬 키워드 목록") # line skip 줄바꿈한다.
print(py_keyword)
#['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

print("명령어 개수: ", len(py_keyword)) #len은 길이 반환 기본함수 # 명령어 개수 : 35


# 낙타체
korScore = 90
matScore = 85
engScore = 80
totalScore = korScore + matScore + engScore

print("총점은 =",totalScore) #총점은 = 255

# 3. 참조변수 : 객체가 저장된 메모리 주소를 참조하는 변수
x = 150 # 150 객체 -> 메모리에 저장 (주소값을 저장)
y = 45.23 #45.23 객체 -> 메모리 저장
y2 = y # 객체 복사 copy, 주소복사
x2 = 150 # 150 객체 -> 메모리 저장 # 객체 -> 메모리 주소 반환한다(동일 객체 생성 안함) 메모리를 효율적으로 사용한다.


# id()로 객체의 주소를 확인한다.
print(x,id(x)) #150 4349853120
print(y,id(y)) #45.23 4369558512 # 주소가 같다면 참조하는 변수가 같다.
print(y2,id(y2)) #45.23 4369558512 # 주소가 같다면 참조하는 변수가 같다.
print(x2,id(x2)) # 150 4349853120 # 값이 같다면 참조하는 변수가 같고 같은 주소값을 가진다. 주소가 같다면 가리키는 객체가 동일하다.


# 변수와 내용(객체) 비교
if (x==150):
    print("x는 150객체를 참조 한다") # True

if (x is x2):
    print("x와 x2의 주소는 같다.") # True










