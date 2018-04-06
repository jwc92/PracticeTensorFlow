class SimpleClass():
	
	def __init__(self,name):
		print("hello" + name)
	def yell(self):
		print("YELLING")


x = SimpleClass('yong')
x.yell()

class ExtendedClass(SimpleClass):
	def __init__(self):
		
		super().__init__('Jose')
		print("EXTEND!")
		
y = ExtendedClass()
y.yell()
