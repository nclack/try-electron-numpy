#include "addon.h"
#include <stdio.h>
#include <Python.h>

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

static int counter;

static napi_value DoSomethingUseful(napi_env env, napi_callback_info info) {
    // Do something useful.

    char buf[1024] = {0};
    snprintf(buf, sizeof(buf), "Hello from addon #%d", counter++);

    napi_value result;
    NAPI_CALL(env,
              napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &result));
    return result;
}

static napi_value onePythonCall(napi_env env, napi_callback_info info) {
    napi_value result;

    Py_SetProgramName(L"try-electron");
    Py_Initialize();
    PyRun_SimpleString("from time import time,ctime\n"
                       "print('Today is', ctime(time()))\n");
    if(Py_FinalizeEx() < 0) {
        // TODO: Handle error.

        NAPI_CALL(env,
                napi_create_string_utf8(env, "Python error", NAPI_AUTO_LENGTH, &result));
        return result;
    }
    NAPI_CALL(env,
            napi_create_string_utf8(env, "Python OK", NAPI_AUTO_LENGTH, &result));

    return result;
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