# instance method
# static method 
# class method






class Calculator:




    def mul(self, a, b):
        return a * b
    
    def summ(self, a , b):
        return a + b
    
    @staticmethod
    def zero_check(b):
        if b == 0 :
            return False
        return True


    def devide(self, a, b):
        if self.zero_check(b):
            return a / b
        return 'devided by zero. not valid'
    





Cal = Calculator()

print(Cal.zero_check( 0))