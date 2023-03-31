from random import*

def generate():
        length = randint(1, 150 + 1)
        print(length)
        
        for i in range(0, length):
                print(uniform(-100, 100), end=" ")
                print(uniform(-75, 75))

        polynomialDegree = randint(1, 15 + 1)
        print(polynomialDegree)

generate()


