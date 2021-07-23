#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "addon.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#define NAPI_CALL(env, call)                                              \
    do {                                                                  \
        napi_status status = (call);                                      \
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
                return NULL;                                              \
            }                                                             \
        }                                                                 \
    } while (0)

///

static int counter;

static napi_value DoSomethingUseful(napi_env env, napi_callback_info info) {
    char buf[1024] = {0};
    snprintf(buf, sizeof(buf), "Hello from addon #%d", counter++);

    napi_value result;
    NAPI_CALL(env,
              napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &result));
    return result;
}

///

static napi_value onePythonCall(napi_env env, napi_callback_info info) {
    napi_value result;

    // 1. Get the filename of to read using numpy.fromfile as an argument

    char filename[1024] = {0};
    {
        size_t     argc = 1;
        napi_value val_filename, this;
        void*      data;
        NAPI_CALL(
            env,
            napi_get_cb_info(env, info, &argc, &val_filename, &this, &data));
        if (argc == 0) {
            napi_throw_error(
                env, NULL, "Expected one argument: No filename provided");
            return NULL;
        }
        size_t sz;
        NAPI_CALL(env,
                  napi_get_value_string_utf8(
                      env, val_filename, filename, sizeof(filename), &sz));
        // FIXME: Check sz is valid (length excluding null)
    }
    printf("filename: %s\n", filename);

    // 2. Invoke numpy.fromfile(filename).
    //    a. Initialize python
    //    b. import numpy
    //    c. call the function
    //    d. get the result as a C numpy array. See: <numpy/arrayobject.h>

    napi_value output_array;
    {
        // FIXME: leaky exception handling
        // FIXME: probably leaky python objects
        Py_Initialize();
        import_array();
        PyObject* numpy = PyImport_ImportModule("numpy");
        if (!numpy) {
            napi_throw_error(env, NULL, "Error importing numpy");
            return NULL;
        }

        PyObject* fromfile = PyObject_GetAttrString(numpy, "fromfile");
        if (!fromfile || !PyCallable_Check(fromfile)) {
            if (PyErr_Occurred()) {
                PyErr_Print();
            }
            napi_throw_error(
                env, NULL, "Error getting function: numpy.fromfile");
            return NULL;
        }

        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(filename));
        PyArrayObject* result =
            (PyArrayObject*)PyObject_CallObject(fromfile, args);
        Py_DECREF(args);
        if (!result || !PyArray_Check(result)) {
            if (PyErr_Occurred())
                PyErr_Print();
            char buf[1024] = {0};
            snprintf(buf, sizeof(buf), "Failed: numpy.fromfile(%s)", filename);
            napi_throw_error(env, NULL, buf);
            return NULL;
        }

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

        {  // Fill in the output array buffer
            void* data;
            NAPI_CALL(env,
                      napi_create_arraybuffer(
                          env, PyArray_NBYTES(result), &data, &output_array));
            memcpy(data, PyArray_DATA(result), PyArray_NBYTES(result));
        }

        Py_XDECREF(fromfile);
        Py_DECREF(numpy);
        Py_FinalizeEx();
    }
    return output_array;
    // 3. Create a node object with the result
    //    a. Create a Float64Array with the pointer.  (type, lifetime?)
    //    b. Create arrays for the shape and strides.
}

napi_value create_addon(napi_env env) {
    napi_value result;
    NAPI_CALL(env, napi_create_object(env, &result));

    // doSomethingUseful

    napi_value exported_function;
    NAPI_CALL(env,
              napi_create_function(env,
                                   "doSomethingUseful",
                                   NAPI_AUTO_LENGTH,
                                   DoSomethingUseful,
                                   NULL,
                                   &exported_function));

    NAPI_CALL(env,
              napi_set_named_property(
                  env, result, "doSomethingUseful", exported_function));

    // onePythonCall

    NAPI_CALL(env,
              napi_create_function(env,
                                   "onePythonCall",
                                   NAPI_AUTO_LENGTH,
                                   onePythonCall,
                                   NULL,
                                   &exported_function));

    NAPI_CALL(env,
              napi_set_named_property(
                  env, result, "onePythonCall", exported_function));

    return result;
}