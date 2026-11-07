from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


# CMake builds the shared library before packaging; the wheel is platform-specific.
setup(distclass=BinaryDistribution)
