#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "addon.h"
#include <Python.h>
#include <numpy/arrayobject.h>  // Q: Why am I getting deprecation warnings?
#include <stdio.h>

//
// Error handling macros
//

#define NAPI_CALL(call, env, ...)                                         \
    do {                                                                  \
        napi_status status = (call)((env), __VA_ARGS__);                  \
        if (status != napi_ok) {                                          \
            const napi_extended_error_info* error_info = NULL;            \
            napi_get_last_error_info((env), &error_info);                 \
            bool is_pending;                                              \
            napi_is_exception_pending((env), &is_pending);                \
            if (!is_pending) {                                            \
                const char* message = (error_info->error_message == NULL) \
                                        ? "empty error message"           \
                                        : error_info->error_message;      \
                napi_throw_error((env), NULL, message);                   \
                goto Error;                                               \
            }                                                             \
        }                                                                 \
    } while (0)

#define EXPECT(expr, ...)                            \
    do {                                             \
        if (!(expr)) {                               \
            char buf[1024] = {0};                    \
            snprintf(buf, sizeof(buf), __VA_ARGS__); \
            napi_throw_error(env, NULL, buf);        \
            goto Error;                              \
        }                                            \
    } while (0)

#define PYEXPECT(expr, ...)                          \
    do {                                             \
        if (!(expr)) {                               \
            if (PyErr_Occurred()) {                  \
                PyErr_Print();                       \
            }                                        \
            char buf[1024] = {0};                    \
            snprintf(buf, sizeof(buf), __VA_ARGS__); \
            napi_throw_error(env, NULL, buf);        \
            goto Error;                              \
        }                                            \
    } while (0)

///

static int counter;

static napi_value DoSomethingUseful(napi_env env, napi_callback_info info) {
    char buf[1024] = {0};
    snprintf(buf, sizeof(buf), "Hello from addon #%d", counter++);

    napi_value result;
    NAPI_CALL(napi_create_string_utf8, env, buf, NAPI_AUTO_LENGTH, &result);
    return result;
Error:
    return NULL;
}

///

static napi_value onePythonCall(napi_env env, napi_callback_info info) {
    // FIXME: multiple calls to onePythonCall doesn't work.
    //        Results in a numpy init exception
    napi_value result;

    // 0. Init python, do imports

    Py_Initialize();
    printf("here %d\n", __LINE__);
    PyObject* numpy = PyImport_ImportModule("numpy");
    printf("here %d %p\n", __LINE__, numpy);
    EXPECT(numpy != NULL, "Could not import numpy");
    printf("here %d\n", __LINE__);
    import_array();
    printf("here %d\n", __LINE__);

    PyObject* numpy_fromfile = PyObject_GetAttrString(numpy, "fromfile");
    PYEXPECT(numpy_fromfile, "Could not get numpy.fromfile");
    PYEXPECT(PyCallable_Check(numpy_fromfile),
             "Expected numpy.fromfile to be a function.");

    // 1. Get the filename of to read using numpy.fromfile as an argument

    char filename[1024] = {0};
    {
        size_t     argc = 1;
        napi_value val_filename, this;
        void*      data;
        NAPI_CALL(
            napi_get_cb_info, env, info, &argc, &val_filename, &this, &data);
        EXPECT(argc > 0, "Expected one argument: No filename provided");

        size_t sz;
        NAPI_CALL(napi_get_value_string_utf8,
                  env,
                  val_filename,
                  filename,
                  sizeof(filename),
                  &sz);
        EXPECT(sz > 0, "Expected a non-empty filename");
    }
    printf("filename: %s\n", filename);

    // 2. Invoke numpy.fromfile(filename).
    //    a. Initialize python
    //    b. import numpy
    //    c. call the function
    //    d. get the result as a C numpy array. See: <numpy/arrayobject.h>

    napi_value output_array;
    {
        // FIXME: probably leaky python objects, esp around exceptions

        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(filename));
        PyArrayObject* result =
            (PyArrayObject*)PyObject_CallObject(numpy_fromfile, args);
        Py_XDECREF(args);
        PYEXPECT(result != NULL, "Failed: numpy.fromfile(\"%s\")", filename);
        PYEXPECT(PyArray_Check(result),
                 "Failed: numpy.fromfile(%s) returned a non-array object.",
                 filename);

        // Inspect the array
        {
            printf("ndim: %d\n", PyArray_NDIM(result));
            char      buf[1024] = {0};
            npy_intp* dims      = PyArray_DIMS(result);
            int       i         = 0;
            char*     c         = buf;
            for (; i < PyArray_NDIM(result) - 1; i++) {
                c += snprintf(c, sizeof(buf) - (c - buf), "%d,", (int)dims[i]);
            }
            snprintf(c, sizeof(c), "%d", (int)dims[i]);
            printf("shape: [%s]\n", buf);
            printf("dtype: %c%c\n",
                   PyArray_DTYPE(result)->byteorder,
                   PyArray_DTYPE(result)->type);
        }

        // 3. Create a node object with the result
        //
        // Q: How to create a TypedArray (float64array)
        // A: napi_create_typed_array
        //
        // Note: The data is copied here. 'napi_create_external_arraybuffer'
        //       can be used to avoid the copy. It's well designed - part of
        //       creating the external buffer is passing a 'finalizer' that
        //       can be used to free the memory. But since we're shutting down
        //       Python within this call, the lifetime of the ndarray is limited
        //       to the duration of the call.
        {
            napi_value arraybuffer;
            void*      data;
            NAPI_CALL(napi_create_arraybuffer,
                      env,
                      PyArray_NBYTES(result),
                      &data,
                      &arraybuffer);
            memcpy(data, PyArray_DATA(result), PyArray_NBYTES(result));

            NAPI_CALL(napi_create_typedarray,
                      env,
                      napi_float64_array,
                      PyArray_SIZE(result),
                      arraybuffer,
                      0,
                      &output_array);

            Py_DECREF(result);
        }
    }

Finalize:
    printf("here %d\n", __LINE__);
    Py_XDECREF(numpy_fromfile);
    Py_XDECREF(numpy);
    Py_FinalizeEx();
    return output_array;

Error:
    output_array = NULL;
    goto Finalize;
}

napi_value create_addon(napi_env env) {
    napi_value result;
    NAPI_CALL(napi_create_object, env, &result);

    // doSomethingUseful

    napi_value exported_function;
    NAPI_CALL(napi_create_function,
              env,
              "doSomethingUseful",
              NAPI_AUTO_LENGTH,
              DoSomethingUseful,
              NULL,
              &exported_function);

    NAPI_CALL(napi_set_named_property,
              env,
              result,
              "doSomethingUseful",
              exported_function);

    // onePythonCall

    NAPI_CALL(napi_create_function,
              env,
              "onePythonCall",
              NAPI_AUTO_LENGTH,
              onePythonCall,
              NULL,
              &exported_function);

    NAPI_CALL(napi_set_named_property,
              env,
              result,
              "onePythonCall",
              exported_function);

    return result;

Error:
    return NULL;
}