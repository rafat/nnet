add_executable(nnettest nnettest.c)

target_link_libraries(nnettest nnetlib m)

add_executable(xor_gdx xor_gdx.c)

target_link_libraries(xor_gdx nnetlib m)

set_target_properties(nnettest xor_gdx
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/test"
)