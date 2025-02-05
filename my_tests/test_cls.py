class MyClass:
    def __new__(cls, value, *args, **kwargs):
        # print(f"__new__ called with cls={cls}, value={value}, args={args}, kwargs={kwargs}")
        # if value < 0:
        #     value = 0  # Modify the argument before creating the instance
        # # Create and return the instance. Essential!
        instance = super().__new__(cls) # or instance = object.__new__(cls) for immutable types
        # instance.value = value
        return instance

    def __init__(self, value, *args, **kwargs):
        print(f"__init__ called with self={self}, value={value}, args={args}, kwargs={kwargs}")
        self.value = value
        self.args = args
        self.kwargs = kwargs

obj = MyClass(-5, "arg1", "arg2", kw1="val1")
print(obj.value)  # Output: 0 (because __new__ modified the value)python