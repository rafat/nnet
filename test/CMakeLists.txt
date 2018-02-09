add_executable(building building.c)

target_link_libraries(building nnetlib m)

add_executable(gene_train gene_train.c)

target_link_libraries(gene_train nnetlib m)

add_executable(gene_test gene_test.c)

target_link_libraries(gene_test nnetlib m)

add_executable(mushroom mushroom.c)

target_link_libraries(mushroom nnetlib m)

add_executable(robot robot.c)

target_link_libraries(robot nnetlib m)

add_executable(soybean soybean.c)

target_link_libraries(soybean nnetlib m)

add_executable(thyroid thyroid.c)

target_link_libraries(thyroid nnetlib m)

set_target_properties(building gene_train gene_test mushroom robot soybean thyroid
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/test"
)