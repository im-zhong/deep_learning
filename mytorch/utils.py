# 2023/9/9
# zhangzhong

import inspect

# 这个函数只应该被类使用
# 所以最好的使用方式是继承


# 貌似改成装饰器更好
class ParametersToAttributes:
    def make_parameters_be_attributes(self, ignore=[]):
        """
        save all arguments in a class's __init__ method as class attributes

        must be called in __init__ method
        """
        # Get a signature object for the passed callable.
        # 只用signature是不够的，因为他只会拿到参数的名字 拿不到值
        # 想要获得值 就需要从frame中拿到
        # sig = inspect.signature(cls.__init__)
        # for name, value in sig.parameters.items():
        #     # 是不是需要去掉self?
        #     if name != 'self':
        #         # sets the named attribute on the given object to the specified value.
        #         setattr(cls, name, value)

        # This line of code gets the previous frame in the call stack
        # frame = inspect.currentframe().f_back
        current_frame = inspect.currentframe()
        if current_frame is not None:
            frame = current_frame.f_back
            assert frame is not None
        else:
            assert False
        # Get information about arguments passed into a particular frame.
        # 'args' is a list of the argument names
        # 'varargs' and 'varkw' are the names of the * and ** arguments or None
        # 'locals' is the locals dictionary of the given frame
        args, _, _, values = inspect.getargvalues(frame)
        for arg in args:
            # filter the self, ignored, and _xxx paramters
            if arg not in set(ignore + ['self']) and not arg.startswith('_'):
                setattr(self, arg, values[arg])
