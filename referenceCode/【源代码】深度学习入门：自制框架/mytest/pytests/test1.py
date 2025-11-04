def decorator(func):
    def wrapper(*args, **kwargs):
        print("调用前")
        result = func(*args, **kwargs)
        print("调用后")
        return result
    return wrapper

@decorator  # 相当于 foo = decorator(foo)
def foo():
    print("foo 被调用")

foo()
