# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:23:21 2019

@author: Junz
"""

import json
from xml.dom.minidom import Document

def traverse_data(json_data,preNode = None):
    # 创建根节点
    if isinstance(json_data,dict):
        dict_data= {}
        #获取所有的key值
        keys = json_data.keys()
        node = doc.createElement("node")
        for var in keys:
            if var != "children":
                node.setAttribute(str(var), str(json_data[var]))
        if preNode != None:
            preNode.appendChild(node)
        else:
            doc.appendChild(node)
        # 创建根元素
        if  json_data.get("children") != None:
            traverse_data(json_data.get("children"),node)

    elif isinstance(json_data,list):
        for element in json_data:
            if element != None and len(element) > 0:
                if isinstance(element,dict):
                    dict_data1 = {}
                    # 获取所有的key值
                    keys1 = element.keys()
                    node = doc.createElement("node")
                    for var1 in keys1:
                        if var1 != "children":
                            print(element[var1])
                            node.setAttribute(str(var1),str(element[var1]))
                    if preNode != None:
                        preNode.appendChild(node)
                    else:
                        doc.appendChild(node)
                    if element.has_key("children") and element.get("children") != None:
                        traverse_data(element.get("children"),node)

src = open('E:/Leaf Sampling/collections/train/1/label/via_project_15Apr2019_23h20m.json')
obj = json.loads(src.read())
doc = Document()
traverse_data(obj)
fp = open("E:/Leaf Sampling/collections/train/1/label/corn.xml", 'w')
doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

