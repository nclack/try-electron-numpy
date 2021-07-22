set(tgt iffy)
add_library(${tgt} SHARED 
    addon.h
    addon.c
    addon-node.c
    )
target_link_libraries(${tgt} Python3::Python Python3::NumPy)
target_include_directories(${tgt} PUBLIC ${CMAKE_JS_INC})
set_target_properties(${tgt} PROPERTIES PREFIX "" SUFFIX ".node")
