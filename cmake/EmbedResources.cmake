include(CMakeParseArguments)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tmp)
add_executable(embed_resources ${CMAKE_CURRENT_LIST_DIR}/EmbedResources.cpp)

function(embed_resources)
    set(SOLO_VARS OUTPUT NAMESPACE)
    cmake_parse_arguments(EMBED_RESOURCES "" "${SOLO_VARS}" "SOURCES" ${ARGN})

    if("${EMBED_RESOURCES_OUTPUT}" STREQUAL "")
        message(WARNING "embed_resources needs an OUTPUT parameter!")
        return()
    endif()

    set(OUTPUT_FILES "")
    foreach(RESOURCE_FILE ${EMBED_RESOURCES_SOURCES})

        # set up input and output variables
        get_filename_component(RESOURCE_PATH "${RESOURCE_FILE}" PATH)
        get_filename_component(RESOURCE_NAME "${RESOURCE_FILE}" NAME)
        set(OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RESOURCE_PATH}")
        set(OUTPUT_FILE "${OUTPUT_PATH}/${RESOURCE_NAME}.cpp")

        # set the build variable
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_PATH}"
            COMMAND ${CMAKE_BINARY_DIR}/embed_resources ${RESOURCE_NAME} ${OUTPUT_FILE} ${EMBED_RESOURCES_NAMESPACE}
            COMMENT "Compiling ${RESOURCE_FILE} to C++ source"
            DEPENDS embed_resources
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_PATH}"
        )

        list(APPEND OUTPUT_FILES "${OUTPUT_FILE}")
    endforeach()

    set(${EMBED_RESOURCES_OUTPUT} ${OUTPUT_FILES} PARENT_SCOPE)
endfunction(embed_resources)
