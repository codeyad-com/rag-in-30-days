class Engine:

    def start():
        return 'Motor started'
    

    def stop():
        return 'Motor stoped'
    




class Car:
    def __init__(self,model, color):
        self.tu5_engine = Engine
        self.model = model
        self.color  = color


    def drive(self):
        return f'Car: {self.model} , {self.tu5_engine.start()}'
    

    def park(self):
        return f'Car: {self.model}, {self.tu5_engine.stop()}'
    





samand = Car( model = 'samand', color = 'white')

print(samand.model)
print(samand.drive())
print(samand.park())
