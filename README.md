# Read a numpy array into an electron app via embedded python

## Prerequisites

- [nvm](https://github.com/creationix/nvm)
- [yarn](https://yarnpkg.com/en/docs/install)
- [python3](https://www.python.org/downloads/)
- [numpy](https://www.numpy.org/)
- [cmake](https://cmake.org/)
- C/C++ compiler

## Building and running

1. `nvm use latest` - This was done w v16.4.2
2. `yarn install`
3. `yarn start`

## What to expect

The first time you run, you'll see an electron window pop up, but in the
terminal you'll find this error message:

```
FileNotFoundError: [Errno 2] No such file or directory: 'data.f64'
```

Create that file! Here's an example in python w numpy:

```python
from numpy import *
a=sin(linspace(0,2*pi,1000))
with open("data.f64","wb") as f:
    f.write(a)
```

The next time you run, check the developer console (open dev tools from the menu
in the electron window).

You'll see this:

```
before python call
ArrayBuffer(8000)
after python call
```

Pretty sweet!

## Notes

* Error handling and memory management on the C side are bad.
* Would be nice to return the correct array subtype
* Would be nice to explore returning dimensional data.  Currently just calling `numpy.fromfile()` which always returns 1-d.
