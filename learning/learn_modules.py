#Crée une fonction fibonacci
def fibonacci(n):
    n1 = 1
    n2 = 1
    fibo = [n1, n2]
    while n>n2:
        n1, n2 = n2, n1+n2
        fibo.append(n2)
    return fibo

def expo(n):
    a = 1
    liste = []
    while a<n:
        liste.append(a)
        a = a*2
    return liste

def ii(stack = 1000,ajout = 500, goal=100000):
    m = 0
    while stack<goal:
        stack = stack*1.05 + ajout

        m+=1
    return f"Il te faudra {m} mois ou {int(m/12)} anées pour atteindre {stack}€"
print(ii(1000, 2000))