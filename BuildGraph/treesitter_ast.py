import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree, Node, Point
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda")
# Code encoder?
# Deepseek encoder?
# Modern Bert?
# + cross atten for problem statement.

# root node, subgraph, problem statement -> root node attr?


def main(INPUTDIR: str, OUTPUTDIR:str):
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    typelist = []

    class MyNode:
        type: str
        name: str
        startpoint: Tuple[int, int]
        endpoint: Tuple[int, int]
        attr: Dict[str, str|List[str]]
        child: List

        def __init__(self, type: str, name: str, startpoint: Tuple[int, int], endpoint: Tuple[int, int], **attr):
            self.type = type
            self.name = name
            self.startpoint = startpoint
            self.endpoint = endpoint
            self.attr = attr
            self.child = []

            self.calllist = []
            self.typelist = []
            self.importlist = []
            self.superclasslist = []
            self.id = -1

        def addchild(self, child):
            self.child.append(child)

        def addattr(self, **attr):
            assert set(attr.keys()).isdisjoint(set(self.attr.keys()))
            self.attr.update(attr)

        def printtree(self, prefix: str=""):
            if self.type not in ["call", "type", "import_from_statement", "import_statement", "future_import_statement"]:
                print(prefix+self.type+" "+self.name)
            for _ in self.child:
                _.printtree(prefix+"    ")

        def is_import(self):
            return self.type in ["import_from_statement", "import_statement", "future_import_statement"]
        
        def is_call(self):
            return self.type in ["call"]
        
        def is_announce(self):
            return self.type in ["type"]
        
        def is_pythonfile(self):
            return self.type in ["python file"]
        
        def is_path(self):
            return self.type in ["directory", "file", "python file"]
        
        def is_classdef(self):
            return self.type in ["class_definition"]
        
        def is_funcdef(self):
            return self.type in ["function_definition"]

        def is_def(self):
            return self.is_classdef() or self.is_funcdef()

        def __repr__(self) -> str:
            return f"{self.type}: {self.name}"


    def extract_code_structure(tree: Tree, filename: str):
        root_node = tree.root_node


        def unichildren(children: List[Node]):
            assert len(children) == 1
            return children[0]

        def point2tuple(point: Point):
            return (point.row+1, point.column+1)

        def node2text(node: Node):
            return node.text.decode("utf-8")
        
        def visit(node: Node, parent: MyNode):
            # Detect the node type and categorize
            if node.type == 'class_definition':
                name = node2text(unichildren(node.children_by_field_name('name')))
                if len(node.children_by_field_name("superclasses"))>0:
                    superclasses = [(node2text(_), point2tuple(_.start_point), point2tuple(_.end_point)) for _ in unichildren(node.children_by_field_name("superclasses")).children[1:-1]]
                else:
                    superclasses = []
                mnode = MyNode(node.type, name, point2tuple(node.start_point), point2tuple(node.end_point), superclasses=superclasses, filename=filename)
                parent.addchild(mnode)
                parent = mnode

            elif node.type == 'function_definition':
                name = node2text(unichildren(node.children_by_field_name('name')))
                mnode = MyNode(node.type, name, point2tuple(node.start_point), point2tuple(node.end_point), filename=filename)
                parent.addchild(mnode)
                parent = mnode

            elif node.type == 'import_from_statement': 
                name = node2text(unichildren(node.children_by_field_name("module_name")))
                mnode = None
                for importmod in node.children_by_field_name("name"):
                    modname = node2text(importmod)
                    mnode = MyNode(node.type, name, point2tuple(node.start_point), point2tuple(node.end_point), namestartpoint=point2tuple(importmod.start_point), nameendpoint=point2tuple(importmod.end_point), modulename=modname)
                    parent.addchild(mnode)
                if mnode is None:
                    return
                parent = mnode

            elif node.type in ['import_statement', 'future_import_statement']:
                for importmod in node.children_by_field_name("name"):
                    name = node2text(importmod)
                    mnode = MyNode(node.type, name, point2tuple(node.start_point), point2tuple(node.end_point), namestartpoint=point2tuple(importmod.start_point), nameendpoint=point2tuple(importmod.end_point))
                    parent.addchild(mnode)
                parent = mnode # assert no child for import node

            elif node.type == 'decorator':
                # Decorator node-> call node.children[1]
                # call may include call
                mnode = MyNode(node.type, "", point2tuple(node.start_point), point2tuple(node.end_point))
                parent.addchild(mnode)
                parent = mnode

            elif node.type == 'call': # will also include class instantiation
                name = node2text(unichildren(node.children_by_field_name("function")))
                mnode = MyNode(node.type, name, point2tuple(node.start_point), point2tuple(node.end_point))
                parent.addchild(mnode)
                # parent = mnode # call does not form hierarchy

            elif node.type == 'type': # parameter dict announce
                name = node2text(node)
                mnode = MyNode(node.type, name, point2tuple(node.start_point), point2tuple(node.end_point))
                parent.addchild(mnode)
                # parent = mnode # call does not form hierarchy

            else:
                typelist.append(node.type)
            for child in node.children:
                visit(child, parent)

        # Start traversing from the root node
        myroot = MyNode("python file", filename, (0, 0), (0, 0))
        visit(root_node, myroot)
        return myroot

    import os.path as osp
    import os


    def readdir(dirname: str)-> MyNode:
        root = MyNode("directory", dirname, (0, 0), (0, 0))
        for subdir in os.listdir(dirname):
            subdir = osp.join(dirname, subdir)
            if osp.isdir(subdir):
                root.addchild(readdir(subdir))
            else:
                if subdir.endswith(".py"):
                    with open(subdir, "r") as f:
                        tree = parser.parse(bytes(f.read(), "utf-8"))
                    root.addchild(extract_code_structure(tree, subdir))
                else:
                    root.addchild(MyNode("file", subdir, (0, 0), (0, 0)))
        return root


    root = readdir(INPUTDIR)

    def builddefdict(root: MyNode, defdict: Dict[Tuple[str, int], MyNode]):
        if root.is_def():
            assert (root.attr["filename"], root.startpoint[0]) not in defdict
            defdict[(root.attr["filename"], root.startpoint[0])] = root

        for child in root.child:
            builddefdict(child, defdict)

    defdict = {}
    builddefdict(root, defdict)
    # print(defdict)



    import jedi
    from jedi.api.classes import Name as JediName

    def is_innermodule(modulepath, rootpath):
        absrootpath: str = osp.abspath(rootpath)
        absmodulepath: str = osp.abspath(modulepath)
        return absmodulepath.startswith(absrootpath)

    def identifycall(root: MyNode, defdict: Dict[Tuple[str, int], MyNode], script: jedi.Script|None):
        if root.is_pythonfile():
            assert script is None
            script =jedi.Script(path=root.name)
        '''
        if root.is_import():
            assert script is not None
            print(root.type, root.name, root.attr, script.path)
            definitions: list[JediName] = script.infer(root.attr["namestartpoint"][0], root.attr["namestartpoint"][1])
            for definition in definitions:
                print(type(definition), definition, definition.full_name, definition.module_path, definition.line, definition.column)
        '''
        if root.is_call():
            assert script is not None
            try:
                definitions: list[JediName] = script.infer(root.startpoint[0], root.startpoint[1])
            except:
                definitions = []
            for definition in definitions:
                if definition.module_path is None:
                    # print(definition.full_name)
                    continue
                # print(definition.full_name, is_innermodule(definition.module_path, "openhands"))
                if True:#is_innermodule(definition.module_path, INPUTDIR):
                    relpath = osp.relpath(definition.module_path).removeprefix("./")
                    if (relpath, definition.line) not in defdict:
                        pass#print(script._orig_path, root.startpoint, relpath, definition.line)
                    else:
                        root.calllist.append(defdict[(relpath, definition.line)])
        if root.is_classdef():
            sc = root.attr["superclasses"]
            for name, startpoint, endpoint in sc:
                try:
                    definitions: list[JediName] = script.infer(startpoint[0], startpoint[1])
                except:
                    definitions = []
                for definition in definitions:
                    if definition.module_path is None:
                        # print(definition.full_name)
                        continue
                    # print(definition.full_name, is_innermodule(definition.module_path, "openhands"))
                    if is_innermodule(definition.module_path, INPUTDIR):
                        relpath = osp.relpath(definition.module_path).removeprefix("./")
                        if (relpath, definition.line) not in defdict:
                            pass #print(script._orig_path, root.startpoint, relpath, definition.line)
                        else:
                            root.superclasslist.append(defdict[(relpath, definition.line)])
        for child in root.child:
            identifycall(child, defdict, script)

    identifycall(root, defdict, None)

    # build graph

    nodelist = []
    def buildnodelist(root: MyNode):
        if root.is_path() or root.is_def():
            root.id = len(nodelist)
            nodelist.append(root)
        for c in root.child:
            buildnodelist(c)
    buildnodelist(root)

    from datasets import Dataset
    import json
    def my_gen():
        for nodeid, n in enumerate(nodelist):
            n: MyNode
            containlist = [nc.id for nc in n.child if nc.id >=0]
            calllist = [ncc.id for nc in n.child if nc.is_call() for ncc in nc.calllist]
            superclasslist = [nc.id for nc in n.superclasslist]
            jsonattr = json.dumps(n.attr)
            sentences = [n.type, n.name, jsonattr]
            embeddings = model.encode(sentences).cpu()
            yield {"id": nodeid, "name": n.name, "type": n.type, "type_emb": embeddings[0], "name_emb": embeddings[1], "attr_emb": embeddings[2], "startpoint": n.startpoint, "endpoint": n.endpoint, "call": calllist, "contain": containlist, "superclasses": superclasslist, "attr": jsonattr}

    data = Dataset.from_generator(my_gen)
    data.save_to_disk(OUTPUTDIR)