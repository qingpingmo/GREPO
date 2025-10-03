"""
parser for python code graph
"""
import json
import random
from enum import Enum, auto
from typing import Any


class NodeType(Enum):
    """
    节点类型
    """
    REPO = auto(),
    PACKAGE = auto(),
    FILE = auto(),
    TEXTFILE = auto(),
    CLASS = auto(),
    ATTRIBUTE = auto(),
    FUNCTION = auto()
    LAMBDA = auto()


class EdgeType(Enum):
    """
    边类型
    """
    CONTAINS = auto(),
    IMPORTS = auto(),
    EXTENDS = auto(),
    IMPLEMENTS = auto(),
    CALLS = auto(),
    REFERENCES = auto(),


class Direction(Enum):
    """
    关系的方向
    """
    IN = auto(),
    OUT = auto()


class CodeGraph:
    """
    CodeGraph表示了由若干类型节点和边组成的程序代码图。
    """

    def __init__(self, nodes, out_edges, in_edges):
        self.nodes = nodes
        self.out_edges = out_edges
        self.in_edges = in_edges
        self.edges = set()

    def get_nodes(self):
        """
        返回所有节点的集合
        """
        return self.nodes.values()

    def get_nodes_by_type(self, node_type: NodeType) -> list:
        """
        获取特定类型的所有节点
        """
        nodes = []
        for node in self.nodes.values():
            if node.get_type() == node_type:
                nodes.append(node)
        return nodes

    def get_nodes_by_type_and_name(self, node_type: NodeType, node_name) -> list:
        """
        通过node_type和node_name获取节点，如果不存在返回空list
        """
        nodes = []
        for node in self.nodes.values():
            if node.get_type() == node_type and node.name == node_name:
                nodes.append(node)
        return nodes

    def get_node_by_id(self, node_id) -> Any:
        """
        通过node_id获取节点，如果不存在返回None
        :param node_id:
        :return:
        """
        if self.nodes.get(node_id) is None:
            return None
        return self.nodes[node_id]

    # def get_random_node(self, node_type: NodeType) -> list | None:
    def get_random_node(self, node_type: NodeType):
        """
        返回指定类型的1个随机节点
        """
        nodes = self.get_nodes_by_type(node_type)
        if len(nodes) == 0:
            return None
        return random.choice(nodes)

    def get_random_nodes(self, node_type: NodeType, k: int) -> list:
        """
        返回最多指定类型的k个随机节点
        """
        nodes = self.get_nodes_by_type(node_type)
        return random.sample(nodes, min(k, len(nodes)))

    def get_out_nodes(self, node_id, edge_type: EdgeType = None) -> list:
        """
        获取node_id对应节点的出边的直接可达的节点
        :param node_id: 目标节点id
        :param edge_type: 目标节点类型
        :return:
        """
        return self.get_related_nodes(node_id, Direction.OUT, edge_type)

    def get_in_nodes(self, node_id, edge_type: EdgeType = None) -> list:
        """
        获取node_id对应节点的入边的直接可达的节点
        :param node_id: 目标节点id
        :param edge_type: 目标节点类型
        :return:
        """
        return self.get_related_nodes(node_id, Direction.IN, edge_type)

    def get_related_nodes(self, node_id, direction: Direction, edge_type: EdgeType = None) -> list:
        """
        返回目标节点相关的节点
        :param node_id: 节点ID
        :param direction: 关联方向
        :param edge_type: 边类型
        :return:
        """
        if self.get_node_by_id(node_id) is None:
            return []

        # 除非指定in，默认返回out
        if direction == Direction.IN:
            edges = self.in_edges.get(node_id)
        else:
            edges = self.out_edges.get(node_id)

        if edges is None:
            return []

        nodes = set()
        for edge in edges:
            if edge_type is None or edge_type == edge.edge_type:
                if direction == Direction.IN:
                    nodes.add(edge.source)
                else:
                    nodes.add(edge.target)

        return list(nodes)

    def nodes_to_dict(self):
        return [node.to_dict() for node in self.nodes.values()]

    def edges_to_dict(self):
        return [edge.to_dict() for edge in self.edges]

    def to_dict(self):
        return {"nodes": self.nodes_to_dict(), "edges": self.edges_to_dict()}

class Repo:
    """
    代码仓节点
    - repo_name: 代码名
    - group_name: 组名
    """

    def __init__(self, node_id, repo_name, codegraph):
        self.node_id = node_id
        # index = repo_name.index('#') # index函数在找不到 # 时会报错，swebench仓库没有这个信息
        # 这个 reponame 是想去掉 base commit 的信息
        if "#" in repo_name:
            repo_name = repo_name.replace('#', '/', 1)
            try: # 针对不统一的命名方式
                self.repo_name, _ = repo_name.split("#")
            except:
                self.repo_name = repo_name
        else:
            self.repo_name = repo_name
        # index = repo_name.find('#')
        # if index != -1:
        #     self.repo_name = repo_name[index + 1:]
        #     self.group_name = repo_name[:index]
        # else:
        #     self.repo_name = repo_name
        self.group_name = ""
        self.codegraph = codegraph

    @staticmethod
    def get_type() -> NodeType:
        return NodeType.REPO

    def node_size(self):
        return len(self.codegraph.nodes)

    def edge_size(self):
        return len(self.codegraph.edges)

    def node_type_size(self):
        return len(set(map(lambda n: n.get_type(), self.codegraph.nodes.values())))

    def edge_type_size(self):
        return len(set(map(lambda e: e.edge_type, self.codegraph.edges)))

    def query_modules(self):
        modules = self.get_modules()
        s = '\n'.join(list(map(lambda x: x.name, modules)))
        return f'仓库{self.repo_name}包含以下模块:\n{s}'

    def get_modules(self):
        contained_ids = self.codegraph.get_out_nodes(self.node_id, EdgeType.CONTAINS)
        contained = list(map(self.codegraph.get_node_by_id, contained_ids))
        return list(filter(lambda n: n.get_type() == NodeType.PACKAGE, contained))

    def query_files(self):
        files = self.get_files()
        s = '\n'.join(list(map(lambda x: x.name, files)))
        return f'仓库{self.repo_name}包含以下文件:\n{s}'

    def get_files(self):
        files = []
        modules = self.get_modules()
        for module in modules:
            files += module.get_files()
        return files

    def query_classes(self):
        classes = self.get_classes()
        s = '\n'.join(list(map(lambda x: x.name, classes)))
        return f'仓库{self.repo_name}包含了以下类:\n{s}'

    def get_classes(self):
        classes = []
        files = self.get_files()
        for file in files:
            classes += file.get_classes()
        return classes

    def __str__(self):
        return self.repo_name

    def __repr__(self):
        return self.repo_name

    def to_dict(self):
        return {"nodeType": NodeType.REPO.name.capitalize(), "id": self.node_id, "repoName": self.repo_name, "groupName": self.group_name}
    
    def get_content(self):
        return self.repo_name


class Package:
    """
    模块节点
    - name: 模块名
    """

    def __init__(self, node_id, name, codegraph):
        self.node_id = node_id
        self.name = name
        self.codegraph = codegraph

    @staticmethod
    def get_type():
        return NodeType.PACKAGE

    def query_files(self):
        files = self.get_files()
        s = '\n'.join(list(map(lambda x: x.name, files)))
        return f'模块{self.name}中包含以下文件:\n{s}'

    def get_files(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_out_nodes(self.node_id, EdgeType.CONTAINS)))

    def query_classes(self):
        classes = self.get_classes()
        s = '\n'.join(list(map(lambda x: x.name, classes)))
        return f'模块{self.name}中包含了以下类:\n{s}'

    def get_classes(self):
        classes = []
        files = self.get_files()
        for file in files:
            classes += file.get_classes()
        return classes

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def to_dict(self):
        return {"nodeType": NodeType.PACKAGE.name.capitalize(), "id": self.node_id, "name": self.name}

    def get_content(self):
        return self.name

class File:
    """
    文件节点
    - name: 文件名
    """

    def __init__(self, node_id, name, path, text, codegraph, clean_text):
        self.node_id = node_id
        self.name = name
        self.path = path
        self.text = text
        self.codegraph = codegraph
        self.clean_text = clean_text

    @staticmethod
    def get_type():
        return NodeType.FILE

    def query_path(self):
        return f'文件的路径是{self.get_path()}'

    def get_path(self):
        module = self.codegraph.get_node_by_id(self.codegraph.get_in_nodes(self.node_id, EdgeType.CONTAINS)[0])
        path = module.name.replace('.', '/')
        return path + "/" + self.name

    def query_imports(self):
        imports = self.get_imports()
        s = '\n'.join(list(map(lambda x: x.name, imports)))
        return f"文件{self.name}引入以下类:\n{s}"

    def get_imports(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_out_nodes(self.node_id, EdgeType.IMPORTS)))

    def query_classes(self):
        classes = self.get_classes()
        s = '\n'.join(list(map(lambda x: x.name, classes)))
        return f'文件{self.name}包含了以下类:\n{s}'

    def get_classes(self):
        return list(map(self.codegraph.get_node_by_id, self.get_classes_ids()))

    def get_classes_ids(self):
        return self.codegraph.get_out_nodes(self.node_id, EdgeType.CONTAINS)

    def query_functions(self):
        functions = self.get_functions()
        s = '\n'.join(list(map(lambda x: x.header, functions)))
        return f'文件{self.name}中包含以下方法:\n{s}'

    def get_functions(self):
        functions = []
        for clazz in self.get_classes():
            functions += clazz.get_functions()
        return functions

    def query_dependent_files(self):
        dependent_files = self.get_dependent_files()
        if len(dependent_files) == 0:
            return f'文件{self.name}不依赖其它文件'
        s = '\n'.join(list(map(lambda x: x.name, dependent_files)))
        return f'文件{self.name}依赖了以下文件:\n{s}'

    def get_dependent_files(self):
        imports = self.get_imports()
        imported_classes = []
        for i in imports:
            imported_classes += self.codegraph.get_nodes_by_type_and_name(NodeType.CLASS, i.name)
        files = list(filter(lambda f: f is not None, map(lambda c: c.get_containing_file(), imported_classes)))
        return files

    def query_dependent_by_files(self):
        dependent_files = self.get_dependent_by_files()
        if len(dependent_files) == 0:
            return f'文件{self.name}没有被其它文件依赖'
        s = '\n'.join(list(map(lambda x: x.name, dependent_files)))
        return f'文件{self.name}被以下文件依赖了:\n{s}'

    def get_dependent_by_files(self):
        contained_classes = self.get_classes_ids()
        files = []
        for c in contained_classes:
            files += list(map(self.codegraph.get_node_by_id, self.codegraph.get_in_nodes(c, EdgeType.IMPORTS)))
        return files

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def to_dict(self):
        return {"nodeType": NodeType.FILE.name.capitalize(), "id": self.node_id,
                "fileName": self.name, "filePath": self.path, "text": self.text,
                "clean_text": self.clean_text}

    def get_content(self):
        filepath = self.path if self.path else ''
        filename = "# Filename: " + filepath + self.name + "\n"
        if self.clean_text:
            return filename + self.clean_text
        elif self.text:
            return filename + self.text # 为了兼容没有经过 clean text 处理的数据
        else:
            return filename

class TextFile:
    """
    文本文件节点，目前包含MD和XML
    - name: 文件名
    - text: 文本内容
    """

    def __init__(self, node_id, name, text, path, codegraph):
        self.node_id = node_id
        self.name = name
        self.text = text
        self.path = path
        self.codegraph = codegraph

    @staticmethod
    def get_type():
        return NodeType.TEXTFILE

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def to_dict(self):
        return {"nodeType": NodeType.TEXTFILE.name.capitalize(), "id": self.node_id, "name": self.name, "text": self.text, "path": self.path}
    
    def get_content(self):
        if self.text:
            return self.name + self.text
        else:
            return self.name

class Class:
    """
    类节点
    - name: 文件名
    - modifiers: 修饰符
    - comment: 注释
    """

    def __init__(self, node_id, name, class_type, comment, text, start_loc, end_loc, col, codegraph, clean_text):
        self.node_id = node_id
        self.name = name
        self.class_type = class_type
        self.comment = comment
        self.text = text
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.col = col
        self.codegraph = codegraph
        self.clean_text = clean_text

    @staticmethod
    def get_type():
        return NodeType.CLASS

    def query_all_superclasses(self):
        s = self.get_all_superclasses()
        if len(s) == 0:
            return f"{self.name}没有继承或实现的类或接口"
        a = '\n'.join(list(map(lambda x: x.name, s)))
        return f'类{self.name}实现或继承了以下类或接口:\n{a}'

    def get_all_superclasses(self):
        superclasses = set(self.get_superclasses()) | set(self.get_interfaces())
        tmp = set()
        for superclass in superclasses:
            tmp |= superclass.get_all_superclasses()
        return superclasses | tmp

    def get_superclass_and_interfaces(self):
        superclass_or_interface_ids = self.get_superclass_and_interface_ids()
        return list(map(self.codegraph.get_node_by_id, superclass_or_interface_ids))

    def get_superclass_and_interface_ids(self):
        return (set(self.codegraph.get_out_nodes(self.node_id, EdgeType.EXTENDS))
                | set(self.codegraph.get_out_nodes(self.node_id, EdgeType.IMPLEMENTS)))

    def get_superclasses(self):
        out = self.codegraph.get_out_nodes(self.node_id, EdgeType.EXTENDS)
        return list(map(self.codegraph.get_node_by_id, out))

    def get_superclass(self):
        out = self.codegraph.get_out_nodes(self.node_id, EdgeType.EXTENDS)
        if len(out) > 0:
            return self.codegraph.get_node_by_id(out[0])
        return None

    def get_superclass_list(self):
        superclass = self.get_superclass()
        if superclass is None:
            return []
        return [superclass] + superclass.get_superclass_list()

    def get_interfaces(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_out_nodes(self.node_id, EdgeType.IMPLEMENTS)))

    def query_all_subclasses(self):
        s = self.get_all_subclasses()
        if len(s) == 0:
            return f"没有类或接口实现或继承了类{self.name}"
        a = '\n'.join(list(map(lambda x: x.name, s)))
        return f'类{self.name}被以下类实现或继承了:\n{a}'

    def get_all_subclasses(self):
        subclasses = set(self.get_subclasses())
        tmp = set()
        for subclass in subclasses:
            tmp |= subclass.get_all_subclasses()
        return subclasses | tmp

    def get_subclasses(self):
        subclass_ids = self.get_subclass_ids()
        return list(map(self.codegraph.get_node_by_id, subclass_ids))

    def get_subclass_ids(self):
        return (set(self.codegraph.get_in_nodes(self.node_id, EdgeType.EXTENDS))
                | set(self.codegraph.get_in_nodes(self.node_id, EdgeType.IMPLEMENTS)))

    def query_functions(self):
        functions = self.get_functions()
        if len(functions) == 0:
            return f'类{self.name}中没有定义函数'
        a = '\n'.join(list(map(lambda m: m.get_simple_signature(), functions)))
        return f"类{self.name}定义了以下函数:\n{a}"

    def get_functions(self):
        return list(filter(lambda n: n.get_type() == NodeType.FUNCTION,
                           list(map(self.codegraph.get_node_by_id,
                                    self.codegraph.get_out_nodes(self.node_id, EdgeType.CONTAINS)))))

    def query_all_functions(self):
        functions = self.get_all_functions()
        if len(functions) == 0:
            return f'类{self.name}中不包含方法'
        a = '\n'.join(list(map(lambda m: m.header, functions)))
        return f"类{self.name}包含了以下方法:\n{a}"

    def get_all_functions(self):
        functions = self.get_functions()
        for superclass in self.get_superclass_list():
            functions += superclass.get_all_functions()
        return functions

    def query_fields(self):
        fields = self.get_attribute()
        if len(fields) == 0:
            return f'类{self.name}中没有定义字段'
        a = '\n'.join(list(map(lambda f: f'{f.name}:{f.field_type}', fields)))
        return f"类{self.name}定义了以下字段:\n{a}"

    def get_attribute(self):
        return list(filter(lambda n: n.get_type() == NodeType.ATTRIBUTE,
                           list(map(self.codegraph.get_node_by_id,
                                    self.codegraph.get_out_nodes(self.node_id, EdgeType.CONTAINS)))))

    def query_containing_file(self):
        file = self.get_containing_file()
        if file is not None:
            return f'类{self.name}所在的文件路径是{file.get_path()}'
        return f'类{self.name}所在的文件无法找到'

    def get_containing_file(self):
        file_ids = self.codegraph.get_in_nodes(self.node_id, EdgeType.CONTAINS)
        if len(file_ids) == 0:
            return None
        file = self.codegraph.get_node_by_id(file_ids[0])
        return file

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def to_dict(self):
        return {"nodeType": NodeType.CLASS.name.capitalize(), "id": self.node_id, "className": self.name,
                "classType": self.class_type, "comment": self.comment, "text": self.text, "startLoc": self.start_loc,
                "endLoc": self.end_loc, "col": self.col, "clean_text": self.clean_text}
    
    def get_content(self):

        if self.clean_text:
            return self.name + self.clean_text
        elif self.text:
            return self.name + self.text # 为了兼容没有经过 clean text 处理的数据
        else:
            return self.name

class Attribute:
    """
    字段节点
    - name: 字段名
    - field_type: 字段名类型
    """

    def __init__(self, node_id, name, attribute_type, comment, text, start_loc, end_loc, col, codegraph):
        self.node_id = node_id
        self.name = name
        self.attribute_type = attribute_type
        self.comment = comment
        self.text = text
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.col = col
        self.codegraph = codegraph

    @staticmethod
    def get_type():
        return NodeType.ATTRIBUTE
    
    # TODO
    # 这个代码可能会有 bug
    # 新增支持数据
    def get_containing_file(self):
        file_ids = self.codegraph.get_in_nodes(self.node_id, EdgeType.CONTAINS)
        if len(file_ids) == 0:
            return None
        file = self.codegraph.get_node_by_id(file_ids[0])
        return file

    def __str__(self):
        return self.name + ":" + self.attribute_type

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {"nodeType": NodeType.ATTRIBUTE.name.capitalize(), "id": self.node_id, "name": self.name,
                "attributeType": self.attribute_type, "comment": self.comment, "text": self.text, "startLoc": self.start_loc,
                "endLoc": self.end_loc, "col": self.col}
    
    def get_content(self):
        # 不确定这个实体的 comment 是否会出现在 text 中
        # 所以先不加
        if self.text:
            return self.text
        else:
            return self.name

class Function:
    """
    方法节点
    - header: 方法头部信息
    - text: 方法文本,包含注释、方法签名和方法体等
    """

    def __init__(self, node_id, name, header, comment, text, start_loc, end_loc, col, codegraph):
        self.node_id = node_id
        self.name = name
        self.header = header
        self.comment = comment
        self.text = text
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.col = col
        self.codegraph = codegraph

    @staticmethod
    def get_type():
        return NodeType.FUNCTION

    def get_header(self):
        return self.header

    def query_containing_file(self):
        return f'方法{self.header}包含在文件{self.get_containing_file().get_path()}中'

    def get_containing_file(self):
        class_ids = self.codegraph.get_in_nodes(self.node_id, EdgeType.CONTAINS)
        if len(class_ids) == 0:
            return None
        class_node_id = class_ids[0]
        return self.codegraph.get_node_by_id(class_node_id).get_containing_file()

    def query_callees(self):
        callees = self.get_callees()
        s = '\n'.join(list(map(lambda x: x.header, callees)))
        return f'函数{self.header}直接调用了以下函数:\n{s}'

    def get_callees(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_out_nodes(self.node_id, EdgeType.CALLS)))

    def get_callee_ids(self):
        return self.codegraph.get_out_nodes(self.node_id, EdgeType.CALLS)

    def query_callers(self):
        callers = self.get_callers()
        s = '\n'.join(list(map(lambda x: x.header, callers)))
        return f'函数{self.header}被以下函数直接调用了:\n{s}'

    def get_callers(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_in_nodes(self.node_id, EdgeType.CALLS)))

    def get_caller_ids(self):
        return self.codegraph.get_in_nodes(self.node_id, EdgeType.CALLS)

    def query_common_callees(self, function1):
        common = self.get_common_callee_ids(function1)
        if len(common) == 0:
            return f"函数{self.header}和函数{function1.header}没有调用同一个函数"
        s = '\n'.join(list(map(lambda x: x.header, common)))
        return f"函数{self.header}和函数{function1.header}调用了{len(common)}个共同的函数，分别是:\n{s}"

    def get_common_callee_ids(self, function1):
        return set(self.get_callee_ids()) & set(function1.get_callee_ids())

    def query_common_callers(self, function1):
        common = self.get_common_caller_ids(function1)
        if len(common) == 0:
            return f"函数{self.header}和函数{function1.header}没有被同一个函数共同调用"
        s = '\n'.join(list(map(lambda x: x.header, common)))
        return f"函数{self.header}和函数{function1.header}被{len(common)}个方法共同调用，分别是:\n{s}"

    def get_common_caller_ids(self, method1):
        return set(self.get_caller_ids()) & set(method1.get_caller_ids())

    def __str__(self):
        return self.header

    def __repr__(self):
        return self.header

    def to_dict(self):
        return {"nodeType": NodeType.FUNCTION.name.capitalize(), "id": self.node_id, "name": self.name,
                "header": self.header, "comment": self.comment, "text": self.text, "startLoc": self.start_loc,
                "endLoc": self.end_loc, "col": self.col}
    
    def get_content(self):

        if self.text:
            return self.text
        else:
            return self.name

class Lambda:
    """
    Lambda节点
    - text: 方法文本,包含注释、方法签名和方法体等
    """

    def __init__(self, node_id, text, start_loc, end_loc, col, codegraph):
        self.node_id = node_id
        self.text = text
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.col = col
        self.codegraph = codegraph

    @staticmethod
    def get_type():
        return NodeType.LAMBDA

    def query_containing_file(self):
        return f'Lambda{self.text}包含在文件{self.get_containing_file().get_path()}中'

    def get_containing_file(self):
        class_ids = self.codegraph.get_in_nodes(self.node_id, EdgeType.CONTAINS)
        if len(class_ids) == 0:
            return None
        class_node_id = class_ids[0]
        return self.codegraph.get_node_by_id(class_node_id).get_containing_file()

    def query_callees(self):
        callees = self.get_callees()
        s = '\n'.join(list(map(lambda x: x.header, callees)))
        return f'Lambda{self.text}直接调用了以下方法:\n{s}'

    def get_callees(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_out_nodes(self.node_id, EdgeType.CALLS)))

    def get_callee_ids(self):
        return self.codegraph.get_out_nodes(self.node_id, EdgeType.CALLS)

    def query_callers(self):
        callers = self.get_callers()
        s = '\n'.join(list(map(lambda x: x.header, callers)))
        return f'Lambda{self.text}被以下函数直接调用了:\n{s}'

    def get_callers(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_in_nodes(self.node_id, EdgeType.CALLS)))

    def get_caller_ids(self):
        return self.codegraph.get_in_nodes(self.node_id, EdgeType.CALLS)

    def get_common_callee_ids(self, method1):
        return set(self.get_callee_ids()) & set(method1.get_callee_ids())

    def get_common_caller_ids(self, method1):
        return set(self.get_caller_ids()) & set(method1.get_caller_ids())

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def to_dict(self):
        return {"nodeType": NodeType.LAMBDA.name.capitalize(), "id": self.node_id, "text": self.text,
                "startLoc": self.start_loc, "endLoc": self.end_loc, "col": self.col}
        
    def get_content(self):

        if self.text:
            return self.text
        else:
            return ''


class Edge:
    """
    边。当前支持的边类型：
    - contains: 不同层级之间的包含关系，比如Repo包含Module，Module包含File，File包含Class，Class包含Field和Method
    - imports: File对Class的引入依赖关系，对应与Java中的import
    - extends: Class之间的继承关系
    - implements: Class之间的实现关系
    - calls: Method之间的调用关系
    """

    def __init__(self, edge_type, source, target):
        self.edge_type = edge_type
        self.source = source
        self.target = target

    def __str__(self):
        return f"{self.source} --[{self.edge_type.name.lower()}]--> {self.target}"

    def to_dict(self):
        return {"edgeType": self.edge_type.name.lower(), "source": self.source, "target": self.target}


def parse(filename):
    """
    从文件解析CodeGraph
    :param filename: 文件名
    :return: CodeGraph对象
    """
    codegraph = CodeGraph({}, {}, {})

    try:
        with open(filename, 'r') as file:
            content = file.read()
            print(f"Graph file opened: {filename}")
    except json.JSONDecodeError as e:
        print(e.msg)
        return

    try:
        data = json.loads(content)
        print(f"Graph data loaded, size: {sizeof_fmt(len(content))}")
    except json.decoder.JSONDecodeError as e:
        print(e.msg)
        return

    for node in data['nodes']:
        node_type = node['nodeType']
        node_id = node['id']
        if node_type.upper() == NodeType.REPO.name:
            name_1 = node.get('repoName')
            name_2 = node.get('name')
            name = name_1 if name_1 else name_2
            codegraph.nodes[node_id] = Repo(node_id, name, codegraph)
        elif node_type.upper() == NodeType.PACKAGE.name:
            codegraph.nodes[node_id] = Package(node_id, node['name'], codegraph)
        elif node_type.upper() == NodeType.FILE.name:
            # 为了兼容不一致的字段
            name_1 = node.get('fileName')
            name_2 = node.get('name')
            name = name_1 if name_1 else name_2
            
            path_1 = node.get('filePath')
            path_2 = node.get('path')
            path = path_1 if path_1 else path_2

            codegraph.nodes[node_id] = File(node_id, name, path, node.get('text'), codegraph, node.get('clean_text'))
        elif node_type.upper() == NodeType.TEXTFILE.name:
            codegraph.nodes[node_id] = TextFile(node_id, node['name'], node['text'], node['path'], codegraph)
        elif node_type.upper() == NodeType.CLASS.name:
            name_1 = node.get('className')
            name_2 = node.get('name')
            name = name_1 if name_1 else name_2
            codegraph.nodes[node_id] = Class(node_id, name, node.get('classType'),
                                             node.get('comment'), node.get('text'), node.get('startLoc'),
                                             node.get('endLoc'), node.get('col'), codegraph, node.get('clean_text'))
        elif node_type.upper() == NodeType.ATTRIBUTE.name:
            attr_1 = node.get('attributeType')
            attr_2 = node.get('type')
            attr = attr_1 if attr_1 else attr_2
            codegraph.nodes[node_id] = Attribute(node_id, node['name'], attr, node.get('comment'),
                                                 node.get('text'), node.get('startLoc'), node.get('endLoc'),
                                                 node.get('col'), codegraph)
        elif node_type.upper() == NodeType.FUNCTION.name:
            codegraph.nodes[node_id] = Function(node_id, node['name'], node['header'], node['comment'],
                                                node.get('text'), node.get('startLoc'), node.get('endLoc'),
                                                node.get('col'), codegraph)
        elif node_type.upper() == NodeType.LAMBDA.name:
            codegraph.nodes[node_id] = Lambda(node_id, node['text'], node.get('startLoc'),
                                              node.get('endLoc'), node.get('col'), codegraph)
        else:
            print("Unkonwn node type: {0}", node_type)

    for edge in data['edges']:
        edge_type = EdgeType[edge['edgeType'].upper()]
        source = edge['source']
        target = edge['target']

        edge = Edge(edge_type, source, target)
        codegraph.edges.add(edge)
        codegraph.out_edges.setdefault(source, set()).add(edge)
        codegraph.in_edges.setdefault(target, set()).add(edge)

    print(f"Graph data parsed, nodes: {len(codegraph.nodes):,}, edges: {len(codegraph.edges):,}")

    return codegraph


def link_repo_to_package(codegraph: CodeGraph):
    repos = codegraph.get_nodes_by_type(NodeType.REPO)
    if len(repos) == 1:
        repo = repos[0]
        packages = codegraph.get_nodes_by_type(NodeType.PACKAGE)
        for package in packages:
            edge = Edge(EdgeType.CONTAINS, repo.node_id, package.node_id)
            codegraph.edges.add(edge)
            codegraph.out_edges.setdefault(repo.node_id, set()).add(edge)
            codegraph.in_edges.setdefault(package.node_id, set()).add(edge)


def serialize(codegraph: CodeGraph):
    return json.dumps(codegraph.to_dict())


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"