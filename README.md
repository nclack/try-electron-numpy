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

# Notes

**Goal**: Interface a `numpy` array with an `electron` app.

**Approach**:

1. Build an `electron` app.
2. Figure out a shared memory interface between `python` and `node`.
   a. Maybe node already has a `mmap` interface?
   b. Otherwise, use an `ffi` to access `mmap`.
3. Figure out how to create/connect to a python process that makes
   the array available via the shared memory channel.

An alternative approach would be to rely on sockets instead of shared-memory. I want to learn about FFI in electron though. Also,
shared-memory may be more performant, though that's speculative. The
shared-memory approach eliminates the possibilty of having the backend and the frontend on different machines.

## Questions

- Q: How to spawn a process from electron? Would like to just communicate with
  another process.

## Notes

### Second impressions

- Got the ffi working and it's pretty straightforward although I still with
  the bindings could be done on the node side.
- Next is a fork in the road:
  - forget ffi and just spawn a python process with a pipe to electron
    - basicaly requires (de)serialization of passed objects
  - integrate python via the ffi
    - not sure if this is the best approach but gives a high degree of control.

### First impressions

- `electron` is amazing.
- `node-ffi` requires node <= 0.10 lts/dubnium
- Going forward, it looks like node wants to rely on `napi`. This looks  
  relatively bad to me, because it is
  - Binding on the c++ side so we have to worry about stability of the
    c++ node api. An FFI binding on the node side would only have to worry
    about the stability of the library being bound, which we have more control over.
  - Prescribes a build system.
  - Is fairly c++ specific. What about other compilable languages? I
    looks like there is `napi-rs` for rust.
