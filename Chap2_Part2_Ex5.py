def isprime(num):
    if num <= 1:
        return False
    for i in range(2, int(num ** 1/2) + 1):
        if num % i == 0:
            return False
    return True

def issumprime(num):
    for i in range(2, num // 2):
        if isprime(i) and isprime(num-i):
            return True
    return False

n = int(input())
for i in range(n):
    num = int(input())
    if issumprime(num):
        print('YES')
    else:
        print('NO')
