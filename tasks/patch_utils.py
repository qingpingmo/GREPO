import re
import os
import ast
from typing import Sequence, TypedDict, Literal, overload, TypeVar, Tuple
from git import Commit

from utils import logger

class DetailedClassFunctionInfo(TypedDict):
    dir_file_name: str
    cls_func_name: str
    type: Literal["class", "function"]
    start_line: int
    end_line: int
    content: str
    
ClassFunctionInfo = Tuple[str, str]

ReturnTypeFuncClassInfo = TypeVar("ReturnTypeFuncClassInfo", bound=(set[str] | dict[str, tuple[Literal["class", "function"], int, int, str]]))


def find_functions_classes(repo_path: str, file_path: str | None, changed_lines: Sequence[int] | None = None, *, detailed: bool = False, add_complete_code_node: bool = False, file_content: str| None = None) -> set[str] | dict[str, tuple[Literal["class", "function"], int, int, str]]:
    """
    Find function and class names in the given old file content based on changed lines.
    Args:
        repo_path (str): The local path of the repository. We need to read the file from the local repo.
        file_path (str): The path to the Python file. If None, return an empty set.
        changed_lines (Sequence[int] | None): A sequence of focused line numbers. If None, all lines are considered. Note: the start is 1, not 0.
        detailed (bool): If True, return detailed information about the modified functions/classes. Otherwise, return a set of function/class names.
        add_complete_code_node: If True, add an additional virtual function called `__complete_code__` to the result. This function contains the complete code of the file.
        file_content (str | None): The content of the file. If None, read the file from the local repo.
    Returns:
        set[str] | dict[str, tuple[Literal["class", "function"], int, int, str]]: A set of touched function/class names, or detailed information. The detailed information is (class_or_function, start_line, end_line, content).
    """
    # print("begin find!!")    
    from utils import num_lines_of_file
    
    def add_complete_code_node_func(returned: ReturnTypeFuncClassInfo, file_content: str) -> ReturnTypeFuncClassInfo:
        if not add_complete_code_node:
            return returned
        if file_path is None:
            return returned
        # add an additional virtual function called `__complete_code__` to the result
        if isinstance(returned, dict):
            num_lines = num_lines_of_file(os.path.join(repo_path, file_path))
            returned["__complete_code__"] = ("function", 1, num_lines, file_content)
            return returned
        elif isinstance(returned, set):
            returned.add("__complete_code__")
            return returned
        else:
            raise TypeError("returned must be either dict or set, but found {}".format(type(returned)))
    if file_content is None:
        if file_path is None:
            return set() if not detailed else dict()
        local_file_path = os.path.join(repo_path, file_path)
        if not os.path.exists(local_file_path):
            logger.error(f"File {file_path} does not exist.")
            return set() if not detailed else dict()
        if not file_path.endswith(".py"):
            logger.info(f"File {file_path} is not a python file, skip the file.")
            return set() if not detailed else dict()
        with open(local_file_path, "r") as f:
            file_content = f.read()
    # print("begin parse!!")
    # Check if the file is empty
    if not file_content:
        logger.warning(f"File {file_path} is empty.")
        return add_complete_code_node_func(set() if not detailed else dict(), file_content)
    
    if changed_lines is None:
        if file_path is None:
            num_lines = file_content.count("\n") + 1
        else:
            num_lines = num_lines_of_file(os.path.join(repo_path, file_path))
        changed_lines = list(range(1, 1 + num_lines)) # all lines are changed
    touched: set[str] | dict[str, tuple[Literal["class", "function"], int, int, str]]
    if not detailed:
        touched = set()
    else:
        touched = dict()

    from tree_sitter import Language, Parser, Tree, Node, Point
    import tree_sitter_python as tspython
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(bytes(file_content, "utf-8"))
    def extract_code_structure(tree: Tree):
        root_node = tree.root_node

        def unichildren(children):
            assert len(children) == 1
            return children[0]

        def point2tuple(point: Point):
            return (point.row+1, point.column+1)

        def node2text(node: Node):
            return node.text.decode("utf-8")
        
        def visit(node: Node, totalname: str):
            # Detect the node type and categorize
            if node.type == 'class_definition':
                name = node2text(unichildren(node.children_by_field_name('name')))
                totalname = totalname + "." + name
                if isinstance(touched, dict):
                    touched[totalname] = ("class", point2tuple(node.start_point), point2tuple(node.end_point), node2text(node))
                else:
                    touched.add(totalname)

            elif node.type == 'function_definition':
                name = node2text(unichildren(node.children_by_field_name('name')))
                totalname = totalname + "." + name
                if isinstance(touched, dict):
                    touched[totalname] = ("function", point2tuple(node.start_point), point2tuple(node.end_point), node2text(node))
                else:
                    touched.add(totalname)

            for child in node.children:
                visit(child, totalname)
        visit(root_node, "")
        return 

    extract_code_structure(tree)
    return add_complete_code_node_func(touched, file_content)

def compare_old_and_new_files(old_file: str | None, new_file: str | None, old_def: set[str], new_def: set[str]) -> list[tuple[None | ClassFunctionInfo, None | ClassFunctionInfo]]:
    """
    This func is to match the old and new functions/classes in the old and new files. The returned list contains tuples (old_funccls, new_funccls). Each could be a tuple (file_name, funccls_name) or None. The None means the function/class is deleted or newly added. Now, because we match func/cls by their names only, so the funccls_name must be the same.
    Args:
        old_file (str | None): The path to the old file. If None, it means the file is newly created.
        new_file (str | None): The path to the new file. If None, it means the file is deleted. Note that the old_file and new_file could not be the same because the git can find the renamed files.
        old_def (set[str]): The set of functions/classes in the old file.
        new_def (set[str]): The set of functions/classes in the new file.
    Returns:
        list[tuple[None | ClassFunctionInfo, None | ClassFunctionInfo]]: A list of tuples containing function/class names.
    """
    return_list: list[tuple[None | ClassFunctionInfo, None | ClassFunctionInfo]] = []
    for func_cls_name in old_def:
        if func_cls_name in new_def:
            # NOTE: now we match two funcs/classes by their names only.
            # There are the same function/class names in the old and new files.
            assert isinstance(old_file, str) and isinstance(new_file, str)
            return_list.append(
                ((old_file, func_cls_name), (new_file, func_cls_name))
            )
        else:
            # The node is not found in the new file.
            assert isinstance(old_file, str)
            return_list.append(
                ((old_file, func_cls_name), None)
            )
    for func_cls_name in new_def:
        if func_cls_name in old_def:
            continue # These nodes have been processed
        # The node is not found in the old file.
        assert isinstance(new_file, str)
        return_list.append(
            (None, (new_file, func_cls_name))
        )
    return return_list

def compare_old_and_new_files_detailed(old_file: str | None, new_file: str | None, old_def: dict[str, tuple[Literal["class", "function"], int, int, str]], new_def: dict[str, tuple[Literal["class", "function"], int, int, str]]) -> list[tuple[None | DetailedClassFunctionInfo, None | DetailedClassFunctionInfo]]:
    """
    This func is to match the old and new functions/classes in the old and new files. The returned list contains tuple (old_funccls, new_funccls). Each could be a dict or None. The None means the function/class is deleted or newly added. Now, because we match func/cls by their names only, so the dict["cls_func_name"] must be the same.
    Args:
        old_file (str | None): The path to the old file. If None, it means the file is newly created.
        new_file (str | None): The path to the new file. If None, it means the file is deleted.
        old_def (set[tuple[str, Literal["class", "function"], int, int, str]]): The set of functions/classes in the old file.
        new_def (set[tuple[str, Literal["class", "function"], int, int, str]]): The set of functions/classes in the new file.
    Returns:
        list[tuple[None | DetailedClassFunctionInfo, None | DetailedClassFunctionInfo]]: A list of tuples containing detailed information about modified functions/classes.
    """
    return_list = []
    for old_func_cls_name, (old_type, old_start_line, old_end_line, old_content) in old_def.items():
        assert isinstance(old_file, str)
        old_node = {
            "dir_file_name": old_file,
            "cls_func_name": old_func_cls_name,
            "type": old_type,
            "start_line": old_start_line,
            "end_line": old_end_line,
            "content": old_content
        }
        if old_func_cls_name in new_def:
            # There are the same function/class names in the old and new files.
            assert isinstance(new_file, str)
            new_node = {
                "dir_file_name": new_file,
                "cls_func_name": old_func_cls_name,
                "type": new_def[old_func_cls_name][0],
                "start_line": new_def[old_func_cls_name][1],
                "end_line": new_def[old_func_cls_name][2],
                "content": new_def[old_func_cls_name][3]
            }
            
            #print("old===", old_node["content"])
            #print("new===", new_node["content"])
            is_equal = (new_node["content"] == old_node["content"])
        else:
            # The node is not found in the new file.
            new_node = None
            is_equal = False
        return_list.append(
            (old_node, new_node, is_equal)
        )
    for new_func_cls_name, (new_type, new_start_line, new_end_line, new_content) in new_def.items():
        if new_func_cls_name in old_def:
            continue # These nodes have been processed
        # The node is not found in the old file.
        old_node = None
        assert isinstance(new_file, str)
        new_node = {
            "dir_file_name": new_file,
            "cls_func_name": new_func_cls_name,
            "type": new_type,
            "start_line": new_start_line,
            "end_line": new_end_line,
            "content": new_content
        }
        return_list.append(
            (old_node, new_node, False)
        )
    return return_list
    

def inner_parse_commit(repo_name: str, base_commitid: str | None, commitid: str, old_files: Sequence[str | None], new_files: Sequence[str | None], repo_root_path: str='repos', *, detailed: bool = False) -> list[tuple[None | ClassFunctionInfo, None | ClassFunctionInfo]] | list[tuple[None | DetailedClassFunctionInfo, None | DetailedClassFunctionInfo]]:
    """
    analyze what functions or classes are modified in the given files and patches.
    Args:
        repo_name (str): The name of the repository. Used to locate the local repository.
        base_commitid (str | None): The commit ID to check out and the commit has been applied. If None, the analysis will based on the empty tree.
        commitid (str): The commit ID to analyze.
        old_files (Sequence[str]): List of original file names. The name could be `None`, which means the file is newly added.
        new_files (Sequence[str]): List of new file names. The name could be `None`, which means the file is deleted.
        repo_root_path (str): The root path of the repository.
        detailed (bool): Whether to return detailed information about the modified functions/classes. If Ture, return a 
    Returns:
        list[tuple[None | ClassFunctionInfo, None | ClassFunctionInfo]] | list[tuple[None | DetailedClassFunctionInfo, None | DetailedClassFunctionInfo]]: A list of tuples containing function/class names or detailed information. The ClassFunctionInfo is a tuple of (file_path, function/class name), and the DetailedClassFunctionInfo is a dictionary containing detailed information about the function/class with keys: dir_file_name, cls_func_name, type, start_line, end_line, content. All the values are string.
    """
    import subprocess
    
    if len(old_files) != len(new_files):
        raise ValueError("old_files and new_files must have the same length")
    
    # Step 1: Parse the old files
    if base_commitid is not None:
        checkout_results = subprocess.check_output(f"cd {repo_root_path}/{repo_name}; git checkout -f {base_commitid}", shell=True)
        print("git checkout feedback:", checkout_results)
        old_defs = []
        for old_file in old_files:
            old_defs.append(find_functions_classes(os.path.join(repo_root_path, repo_name), old_file, detailed=detailed, add_complete_code_node=True)) # provide changed_lines are None so we detect all the functions/classes in the file
    else:
        old_defs = []

    # Step 2: Parse the new files
    checkout_results = subprocess.check_output(f"cd {repo_root_path}/{repo_name}; git checkout -f {commitid}", shell=True)
    print("git checkout feedback:", checkout_results)
    new_defs = []
    for new_file in new_files:
        new_defs.append(find_functions_classes(os.path.join(repo_root_path, repo_name), new_file, detailed=detailed, add_complete_code_node=True)) # provide changed_lines are None so we detect all the functions/classes in the file
    
    # Step 3: Compare the old and new files
    results = []
    for old_def, new_def, old_file, new_file in zip(old_defs, new_defs, old_files, new_files):
        if isinstance(old_def, dict) and isinstance(new_def, dict):
            assert detailed
            results.append(compare_old_and_new_files_detailed(old_file, new_file, old_def, new_def))
        elif isinstance(old_def, set) and isinstance(new_def, set):
            assert not detailed
            results.append(compare_old_and_new_files(old_file, new_file, old_def, new_def))
        else:
            raise TypeError("old_def and new_def must be the same type and be either dict or set, but found {} and {}".format(type(old_def), type(new_def)))
    # Step 4: Return the results
    return results


import subprocess
def apply_patch(original, patch):
    with open("tmpoutfile", "w") as f:
        f.write(original)
        f.flush()
    with open("tmppatch.diff", "w") as f:
        f.write(patch)
        f.flush()
    subprocess.call(args=["patch", "tmpoutfile", "tmppatch.diff"])
    with open("tmpoutfile", "r") as f:
        return f.read()

def parse_patch(repo_root_path:str, repo_name: str, file_names, file_contents: list[str], file_patches: list[str], *, detailed: bool = False) -> list[tuple[None | ClassFunctionInfo, None | ClassFunctionInfo]] | list[tuple[None | DetailedClassFunctionInfo, None | DetailedClassFunctionInfo]]:    
    if len(file_contents) != len(file_patches):
        raise ValueError("file_contents and file_patches must have the same length")
    
    # Step 1: Parse the old files
    old_defs = []
    for old_file in file_contents:
        old_defs.append(find_functions_classes(os.path.join(repo_root_path, repo_name), None, file_content=old_file, detailed=detailed, add_complete_code_node=True)) # provide changed_lines are None so we detect all the functions/classes in the file

    new_defs = []
    for old_file, patch in zip(file_contents, file_patches):
        new_file = apply_patch(old_file, patch)
        print("////=old", old_file)
        print("////=patch", patch)
        print("////=new", new_file)
        try:
            new_defs.append(find_functions_classes(os.path.join(repo_root_path, repo_name), None, file_content=new_file, detailed=detailed, add_complete_code_node=True)) # provide changed_lines are None so we detect all the functions/classes in the file
        except:
            new_defs.append(None)
    # print("old-new defs", old_defs[0].keys(), new_defs[0].keys())
    # Step 3: Compare the old and new files
    results = []
    for old_def, new_def, file_name in zip(old_defs, new_defs, file_names):
        if new_def is None:
            continue
        if isinstance(old_def, dict) and isinstance(new_def, dict):
            assert detailed
            results.append(compare_old_and_new_files_detailed(file_name, file_name, old_def, new_def))
        elif isinstance(old_def, set) and isinstance(new_def, set):
            assert not detailed
            results.append(compare_old_and_new_files(file_name, file_name, old_def, new_def))
        else:
            raise TypeError("old_def and new_def must be the same type and be either dict or set, but found {} and {}".format(type(old_def), type(new_def)))
    # Step 4: Return the results
    return results


def parse_commit(repo_name: str, base_commit: Commit | None, commit: Commit, repo_root_path: str = "repos", *, detailed: bool = False):
    if base_commit is None:
        from git import NULL_TREE
        base_commit_with_null = NULL_TREE
    else:
        base_commit_with_null = base_commit
    # get the file change relation of the commit
    diffs = commit.diff(base_commit_with_null)
    
    old_files = []
    new_files = []
    for diff in diffs:
        old_files.append(diff.a_path)
        new_files.append(diff.b_path)
    return inner_parse_commit(
        repo_name=repo_name,
        base_commitid=base_commit.hexsha if base_commit is not None else None,
        commitid=commit.hexsha,
        old_files=old_files,
        new_files=new_files,
        repo_root_path=repo_root_path,
        detailed=detailed
    )