class Animal:
    def __init__(self,name , color , age ):
        self.name = name
        self.color = color
        self.age = age





    def bark(self):
        if self.name == 'cat':
            return 'meow meow'
        else:
            return 'bark bark'
        
    
    




# class -> object



alex = Animal('dog','brown',5)
fandoogh = Animal('dog','black',3)
lucy = Animal('cat','black-white',0.6)

print(alex.name, alex.age , alex.color , alex.bark())
print(fandoogh.name, fandoogh.age , fandoogh.color , fandoogh.bark())
print(lucy.name, lucy.age , lucy.color , lucy.bark())