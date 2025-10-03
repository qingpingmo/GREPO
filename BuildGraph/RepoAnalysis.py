import git.compat
import git.diff
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree, Node, Point
from typing import Dict, List, Tuple, Literal
import os.path as osp
import os
import git
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from git.diff import Diff
import subprocess
import argparse


NodeType = Literal["directory", "file", "python file", "class def", "func def"]

nodetypedict = {type_name: index for index, type_name in enumerate(NodeType.__args__)}




class MyNode:
    id: int
    type: NodeType
    path_file_name: str
    cls_func_name: str | None # for dir, file, python file, it is None
    lines = None | dict[str, tuple[int, int]] # only for func or cls. {commitid: (start, end)} the lines can be changed from different commits
    attr: Dict[str, str|List[str]]
    child: List
    start_commit: str
    end_commit: str | None
    def __init__(self, id: int, type: NodeType, path_file_name: str, cls_func_name: str | None, lines: dict[str, tuple[int, int]] | None, start_commit: str, end_commit: str ="none", previous: list | None = None, superclasses: list | None = None, **attr):
        self.type = type
        self.path_file_name = path_file_name
        self.cls_func_name = cls_func_name
        self.lines = lines
        self.attr = attr
        self.child = []
        self.calllist = []
        self.ncalllist = []
        self.typelist = []
        self.importlist = []
        self.superclasslist = []
        self.start_commit = start_commit
        self.end_commit = end_commit
        self.superclasses = [] if superclasses is None else superclasses
        self.id = id
        self.previous = previous if previous is not None else []

    def addchild(self, child):
        self.child.append(child)
    
    def addattr(self, **attr):
        assert set(attr.keys()).isdisjoint(set(self.attr.keys()))
        self.attr.update(attr)
    
    def is_pythonfile(self):
        return self.type in ["python file"]
    
    def set_end_recursive(self, commid_id):
        self.end_commit = commid_id
        nodelist = [_ for _ in self.child]
        while len(nodelist) > 0:
            curr = nodelist.pop(-1)
            curr.end_commit = commid_id
            for c in curr.child:
                nodelist.append(c)

    def is_path(self):
        return self.type in ["directory", "file", "python file"]
    
    def is_classdef(self):
        return self.type in ["class def"]
    
    def is_funcdef(self):
        return self.type in ["func def"]
    
    def is_def(self):
        return self.is_classdef() or self.is_funcdef()
    
    def __repr__(self) -> str:
        return f"id: {self.id} {self.type}: {self.name}_{self.version}"

def readcode(INPUTDIR: str, commitid: str="", centerfile: list[str] = None, defdict: dict = None, line2defdict: dict = None, pathdict: dict = None, nodelist=None):
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    if line2defdict is None:
        line2defdict = {}
    if defdict is None:
        defdict = {}
    if pathdict is None:
        pathdict = {}
    if nodelist is None:
        nodelist = []

    def buildnode(type: str, path_file_name: str, cls_func_name: str, lines: Tuple[int, int], **kwargs) -> MyNode:
        nodelist.append(MyNode(len(nodelist), type, path_file_name, cls_func_name, lines, start_commit=commitid, **kwargs))
        root: MyNode = nodelist[-1]
        if root.is_def():
            line2defdict[(root.path_file_name, root.lines[0])] = root
            if (root.path_file_name, root.cls_func_name) in defdict:
                root.previous.append(defdict[(root.path_file_name, root.cls_func_name)])
            defdict[(root.path_file_name, root.cls_func_name)] = root
        if root.is_path():
            if root.path_file_name in pathdict:
                pathdict[root.path_file_name].set_end_recursive(commitid)
                root.previous.append(pathdict[root.path_file_name])
            pathdict[root.path_file_name] = root
        return nodelist[-1]

    def extract_code_structure(tree: Tree, filename: str):
        root_node = tree.root_node

        def unichildren(children: List[Node]):
            assert len(children) == 1
            return children[0]

        def point2tuple(point: Point):
            return (point.row+1, point.column+1)

        def node2text(node: Node):
            return node.text.decode("utf-8")
        
        def visit(node: Node, parent: MyNode, totalname: str):
            # Detect the node type and categorize
            if node.type == 'class_definition':
                name = node2text(unichildren(node.children_by_field_name('name')))
                if len(node.children_by_field_name("superclasses"))>0:
                    superclasses = [(node2text(_), point2tuple(_.start_point), point2tuple(_.end_point)) for _ in unichildren(node.children_by_field_name("superclasses")).children[1:-1]]
                else:
                    superclasses = []
                totalname = totalname + "." + name
                mnode = buildnode("class def", filename, totalname, point2tuple(node.start_point), superclasses=superclasses, code=node2text(node))
                # print(mnode.superclasses)
                parent.addchild(mnode)
                parent = mnode

            elif node.type == 'function_definition':
                name = node2text(unichildren(node.children_by_field_name('name')))
                totalname = totalname + "." + name
                mnode = buildnode("func def", filename, totalname, point2tuple(node.start_point), code=node2text(node))
                parent.addchild(mnode)
                parent = mnode

            elif node.type == 'call':
                parent.ncalllist.append((point2tuple(node.start_point), point2tuple(node.end_point)))

            for child in node.children:
                visit(child, parent, totalname)

        with open(filename, "r") as f:
            fullcode = f.read()
        # Start traversing from the root node
        myroot = buildnode("python file", filename, "", (0, 0), code=fullcode)
        visit(root_node, myroot, "")
        return myroot

    def readfile(filename: str)-> MyNode:
        if filename.endswith(".py"):
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    try:
                        tree = parser.parse(bytes(f.read(), "utf-8"))
                    except:
                        return buildnode("file", filename, "", (0, 0))        
                    return extract_code_structure(tree, filename)
            else:
                return buildnode("file", filename, "", (0, 0))
        else:
            return buildnode("file", filename, "", (0, 0))

    def readdir(dirname: str)-> MyNode:
        root = buildnode("directory", dirname, "", (0, 0))
        for subdir in os.listdir(dirname):
            if subdir.startswith("."):
                continue
            subdir = osp.join(dirname, subdir)
            if osp.isdir(subdir):
                root.addchild(readdir(subdir))
            else:
                root.addchild(readfile(subdir))
        return root


    def buildpath_recur(root: MyNode):
        if osp.dirname(root.path_file_name) in pathdict:
            pathdict[osp.dirname(root.path_file_name.rstrip("/"))].addchild(root)
        else:
            tnode = buildnode("directory", osp.dirname(root.path_file_name.rstrip("/")), "", (0, 0))
            tnode.addchild(root)
            buildpath_recur(tnode)
        
    if centerfile is None:
        roots = [readdir(INPUTDIR)]
    else:
        roots = []
        '''
        for a_path, b_path, patch in centerfile:
            print("**", a_path, b_path)
        '''
        for a_path, b_path, patch in centerfile:
            if a_path is not None:
                if a_path in pathdict:
                    pathdict[a_path].set_end_recursive(commitid)
                else:
                    pass
                    #assert patch is None, f"fail {a_path} {b_path}"
                '''
                from tasks.patch_utils import parse_patch
                if patch is not None:
                    # print("==", a_path, pathdict[a_path].type)
                    patchout = parse_patch("", "", [a_path], [pathdict[a_path].attr["code"]], [patch], detailed=True)[0]
                    # print(a_path, b_path, patch)
                    if len(patchout) == 0:
                        pass
                    else:
                        patchout = patchout
                        for old_node, new_node, is_equal in patchout:
                            print(None if old_node is None else old_node["cls_func_name"], None if new_node is None else new_node["cls_func_name"], is_equal)
                        exit()
                '''
            if b_path is not None:
                roots.append(readfile(b_path))
        for root in roots:
            buildpath_recur(root)
    import jedi
    from jedi.api.classes import Name as JediName

    def get_containing_class(node: MyNode, defdict: Dict) -> MyNode | None:
        
        if node.is_classdef():
            return node
        
        if node.cls_func_name and '.' in node.cls_func_name:
            parts = node.cls_func_name.split('.')
            if len(parts) >= 2:
                
                class_name = '.'.join(parts[:-1])  
                for (path, name), class_node in defdict.items():
                    if path == node.path_file_name and name == class_name and class_node.is_classdef():
                        return class_node
        return None

    def is_subclass(child_class: MyNode, parent_class: MyNode) -> bool:
        
        if not child_class or not parent_class:
            return False
        
        
        def check_inheritance(cls: MyNode, target: MyNode, visited: set) -> bool:
            if cls.id in visited:
                return False
            visited.add(cls.id)
            
            if cls == target:
                return True
                
            
            for superclass in cls.superclasslist:
                if check_inheritance(superclass, target, visited):
                    return True
            return False
        
        return check_inheritance(child_class, parent_class, set())

    def find_or_create_inherited_method(current_class: MyNode, target_method: MyNode, 
                                      line2defdict: Dict, nodelist: list, commitid: str, defdict: Dict) -> MyNode | None:
        
        if not current_class or not target_method:
            return None
            
        
        method_name = target_method.cls_func_name.split('.')[-1]  
        inherited_name = f"{current_class.cls_func_name}.{method_name}"
        
        
        key = (current_class.path_file_name, inherited_name)
        for (path, name), node in defdict.items():
            if path == current_class.path_file_name and name == inherited_name:
                return node
        
        
        inherited_method = MyNode(
            id=len(nodelist),
            type=target_method.type,
            path_file_name=current_class.path_file_name,
            cls_func_name=inherited_name,
            lines=None,  
            start_commit=commitid,
            end_commit="none",
            code=f"# Inherited from {target_method.cls_func_name}"
        )
        
        nodelist.append(inherited_method)
        
        defdict[(current_class.path_file_name, inherited_name)] = inherited_method
        current_class.addchild(inherited_method)
        
        
        inherited_method.calllist.append(target_method)
        
        return inherited_method

    def identifycall(root: MyNode, line2defdict: Dict[Tuple[str, int], MyNode], script: jedi.Script|None, nodelist: list, commitid: str, defdict: Dict):
        if root.is_pythonfile():
            assert script is None
            script =jedi.Script(path=root.path_file_name)
        if root.is_def():
            assert script is not None
            for startpoint, endpoint in root.ncalllist:
                try:
                    definitions: list[JediName] = script.infer(startpoint[0], startpoint[1])
                except:
                    definitions = []
                for definition in definitions:
                    if definition.module_path is None:
                        # print(definition.full_name)
                        continue
                    if True:
                        relpath = osp.relpath(definition.module_path).removeprefix("./")
                        if (relpath, definition.line) not in line2defdict:
                            pass#print(list(line2defdict.keys()), relpath, definition.line)
                        else:
                            target_node = line2defdict[(relpath, definition.line)]
                            root.calllist.append(target_node)
                            
                            
                            if target_node.is_def() and root.is_def():
                               
                                current_class = get_containing_class(root, defdict)
                                target_class = get_containing_class(target_node, defdict)
                                if current_class and target_class and current_class != target_class:
                                   
                                    if is_subclass(current_class, target_class):
                                        
                                        inherited_method = find_or_create_inherited_method(
                                            current_class, target_node, line2defdict, nodelist, commitid, defdict
                                        )
                                        if inherited_method and inherited_method != target_node:
                                            root.calllist.append(inherited_method)

        if root.is_classdef():
            sc = root.superclasses
            
            for name, startpoint, endpoint in sc:
                try:
                    definitions: list[JediName] = script.infer(startpoint[0], startpoint[1])
                except:
                    definitions = []
                for definition in definitions:
                    if definition.module_path is None:
                        # print(definition.full_name)
                        continue
                    if True:
                        relpath = osp.relpath(definition.module_path).removeprefix("./")
                        if (relpath, definition.line) not in line2defdict:
                            pass #print(script._orig_path, root.startpoint, relpath, definition.line)
                        else:
                            root.superclasslist.append(line2defdict[(relpath, definition.line)])
            # root.attr.pop("superclasses")

        for child in root.child:
            identifycall(child, line2defdict, script, nodelist, commitid, defdict)

    for root in roots:
        identifycall(root, line2defdict, None, nodelist, commitid, defdict)



def savetodata(nodelist: list[MyNode], REPO_PATH: str, OUTPUTDIR: str):
    # build graph
    from datasets import Dataset
    import json
    from datasets import Sequence, Value
    import numpy as np
    def my_gen():
        for nodeid, n in enumerate(nodelist):
            containlist = np.array([-1] + list(set([nc.id for nc in n.child if nc.id >=0])))[1:]
            calllist = np.array([-1] + list(set([nc.id for nc in n.calllist])))[1:]
            superclasslist = np.array([-1] + list(set([nc.id for nc in n.superclasslist])))[1:]
            previouslist = np.array([-1] + list(set([nc.id for nc in n.previous])))[1:]
            jsonattr = json.dumps(n.attr)
            ret = {
                "id": n.id, 
                "path": n.path_file_name.removeprefix(REPO_PATH), 
                "name": n.cls_func_name,
                "type": nodetypedict[n.type], 
                "attr": jsonattr, 
                "call": calllist, 
                "contain": containlist, 
                "superclasses": superclasslist, 
                "start_commit": n.start_commit, 
                "end_commit": n.end_commit,
                "previous": previouslist
                }
            
            if n.is_funcdef() or n.is_classdef():
                if identifyfunc(n):
                    ret["modified"] = True
                else:
                    ret["modified"] = False
            else:
                ret["modified"] = False
            yield ret
    nodelist = [_ for _ in my_gen()]
    data = Dataset.from_list(nodelist)
    data.save_to_disk(OUTPUTDIR)

def identifyfunc(node):
    if not (node.is_funcdef() or node.is_classdef()):
        return None
    if not node.previous:
        return None
    prev_node = node.previous[-1]
    prev_code = prev_node.attr.get("code") if prev_node else None
    curr_code = node.attr.get("code")
    return prev_code != curr_code

from commit_utils import CommitDAGAnalyzer
import git
def main(repo_path, repo_name):
    print("main")
    MAIN_BRANCH_NAME = "main"
    try:
        print(subprocess.check_output(f"cd {repo_path}; git checkout -f {MAIN_BRANCH_NAME}", shell=True))
    except Exception:
        try:
            MAIN_BRANCH_NAME = "master"
            print(subprocess.check_output(f"cd {repo_path}; git checkout -f {MAIN_BRANCH_NAME}", shell=True))
        except Exception:
            try:
                MAIN_BRANCH_NAME = "pre-commit-ci-update-config"
                print(subprocess.check_output(f"cd {repo_path}; git checkout -f {MAIN_BRANCH_NAME}", shell=True))
            except Exception:
                MAIN_BRANCH_NAME = "develop2"
                print(subprocess.check_output(f"cd {repo_path}; git checkout -f {MAIN_BRANCH_NAME}", shell=True))

    analyzer = CommitDAGAnalyzer(repo_name)
    sha_path = analyzer.get_longest_path()
    repo = git.Repo(repo_path)

    commits = []
    commits.append((sha_path[0], None))
    
    for i in range(1, len(sha_path)):
        diff = repo.commit(sha_path[i-1]).diff(sha_path[i], create_patch=True)
        commit_files = []
        for d in diff:
            d: git.diff.Diff
            a_path = None if d.a_path is None else osp.join(repo_path, d.a_path)
            b_path = None if d.b_path is None else osp.join(repo_path, d.b_path)
            if not((a_path is not None and a_path.endswith(".py")) or (b_path is not None and b_path.endswith(".py"))):
                patch = None
            else:
                try:
                    patch = d.diff.decode(git.compat.defenc) if isinstance(d.diff, bytes) else d.diff
                except:
                    patch = None
            commit_files.append((a_path, b_path, patch))
        commits.append((sha_path[i], commit_files))
    print("#snapshot", len(commits))
    # Dictionary to store commit dates 
    line2defdict = {}
    pathdict = {}
    defdict = {}
    nodelist = []
    for i, (commit, commit_files) in enumerate(commits):
        try:
            subprocess.check_output(f"cd {repo_path}; git checkout -f {commit}", shell=True)
            print(commit, "success", flush=True)
        except:
            print("commit", commit, "failed")
            continue
        if i==0:
            readcode(repo_path, commit, None, defdict, line2defdict, pathdict, nodelist)
        else:
            readcode(repo_path, commit, commit_files, defdict, line2defdict, pathdict, nodelist)
        # sprint(commit, commit.committed_date)
        # print(d)
    
    savetodata(nodelist, repo_path, osp.join("savedata", repo_path))
    return None


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    args = parser.parse_args()
    repo_path = os.path.join("repos", args.name)  # Replace with your repository path
    main(repo_path, args.name)