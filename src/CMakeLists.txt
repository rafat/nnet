include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCE_FILES    			netint.c
					matrix.c
					netdata.c
					nnet.c
                    )

set(HEADER_FILES    			netint.h
					matrix.h
					netdata.h
					nnet.h
                    )

add_library(nnetlib STATIC ${SOURCE_FILES} ${HEADER_FILES})
        
set_property(TARGET nnetlib PROPERTY FOLDER "lib")

target_include_directories(nnetlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})