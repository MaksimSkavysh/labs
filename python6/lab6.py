class BoundedMeta(type):
    _instances_num = 0

    def __call__(cls, *args, **kw):
        if cls._max_instance_count is None:
            return
        if cls._instances_num >= cls._max_instance_count:
            raise TypeError('Max instances amount reached')
        cls._instances_num = cls._instances_num + 1

    def __new__(mcs, *args, max_instance_count=1):
        obj = super().__new__(mcs, *args)
        obj._max_instance_count = max_instance_count
        return obj


# class C(metaclass=BoundedMeta, max_instance_count=2):
#     pass
#
# c1 = C()
# c2 = C()
#
# try:
#     c3 = C()
# except TypeError:
#     print('everything works fine!')
# else:
#     print('something goes wrong!')


class BoundedBase:
    _instances_num = 0

    def __new__(cls, *args):
        if cls.get_max_instance_count() is None:
            return
        if cls._instances_num >= cls.get_max_instance_count():
            raise TypeError('Max instances amount reached')
        cls._instances_num = cls._instances_num + 1
        obj = super().__new__(cls)
        return obj


class D(BoundedBase):
    @classmethod
    def get_max_instance_count(cls):
        return 1

d1 = D()

try:
    d2 = D()
except TypeError:
    print('everything works fine!')
else:
    print('something goes wrong!')

