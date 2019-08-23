add_executable(raja-sdp-cpu "main.cu")
add_executable(raja-sdp-gpu "main.cu")
set_target_properties(raja-sdp-gpu PROPERTIES COMPILE_FLAGS "-DGPU=On")
