class SimpleClass():
    def __init__(self, name):
        print("hello " + name)
    
    def yell(self):
        print('Yelling')

class ExtendedClass(SimpleClass):

    def __init__(self):
        super().__init__('Andy')
        print('Extend')


def main():

    y = ExtendedClass()
    y.yell()


if __name__ == '__main__':
    main()