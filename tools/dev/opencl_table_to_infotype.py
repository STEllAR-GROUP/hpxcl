#!/usr/bin/env python

import sys

from xml.dom import minidom

if len(sys.argv) < 2:
    print("Expected input path!")
    exit(1)

print("Opening '" + sys.argv[1] + "' ...")

xmldoc = minidom.parse(sys.argv[1])

tables = xmldoc.getElementsByTagName("tbody");

if len(tables) < 1:
    print("Error: Document does not contain a <tbody> tab.")
    exit(1)

print("" + str(len(tables)) + " Table(s) found. Using table #0 ...")


def strip_text(text):
    text = text.strip()
    #text = text.rstrip('-')
    return text

def get_node_text(node):

    text = []

    for childNode in node.childNodes:
        if childNode.nodeType == childNode.TEXT_NODE:
            text.append(strip_text(childNode.data))
            continue
        text.append(get_node_text(childNode))

    return ' '.join(join_newlines(text))

def join_newlines(texts):
    output = []

    while len(texts) > 0:

        current = texts.pop(0).strip()

        if len(current) == 0:
            continue

        if len(output) == 0:
            output.append(current)
            continue
        
        last = len(output) - 1

        if output[last].endswith('-'):
            #print("joining: '" + output[last] + "' and '" + current + "'")
            output[last] = output[last][:-1].strip() + current
        else:
            output.append(current) 

    return output

def read_table(table):

    result = []

    rows = table.childNodes
    
    for row in rows:

        result_row = []

        if row.nodeType != row.ELEMENT_NODE:
            continue
        if row.tagName != "tr":
            continue 
        
        columns = row.childNodes
        
        for column in columns:
            if column.nodeType != column.ELEMENT_NODE:
                continue
            if column.tagName != "td":
                continue 
            text = get_node_text(column)
            result_row.append(text)
            if len(result_row) >= 2:
                break
            
        result.append(result_row)


    return result



table = read_table(tables[0])

max_name_width = 0

for row in table:
    names = row[0].split(' ')
    for name in names:
        max_name_width = max(max_name_width, len(name))


for row in table:
    names = row[0].split(' ')
    datatype = row[1]
    if datatype == "char[]":
        datatype = "std::string"
    elif datatype.endswith("[]"):
        datatype = "std::vector<" + datatype[:-2] + ">"
    datatype = datatype.replace("size_t", "std::size_t")
    for name in names:
        name = name + ","
        while len(name) < max_name_width + 1:
            name = name + " "
        print("HPX_OPENCL_DETAIL_INFO_TYPE_DEVICE( " + name + " " + datatype + " )")


