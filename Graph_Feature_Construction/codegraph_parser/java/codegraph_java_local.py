"""
parser for Java code graph
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
    FIELD = auto(),
    METHOD = auto()


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


current_repo = ""


class Repo:
    """
    代码仓节点
    - path: 代码仓路径
    """

    def __init__(self, node_id, path, codegraph):
        self.node_id = node_id
        self.repo_name = None
        self.path = path
        if path is not None:
            # self.path = path[path.rfind("/") + 1:].replace('#', '/')
            if "#" in path:
                self.repo_name = path.split("/")[-1]
            else:
                self.repo_name = path
        else:
            self.path = current_repo
            self.repo_name = current_repo # 空的仓库名
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
        modules = self.get_packages()
        s = '\n'.join(list(map(lambda x: x.name, modules)))
        return f'仓库{self.path}包含以下模块:\n{s}'

    def get_packages(self):
        contained_ids = self.codegraph.get_out_nodes(self.node_id, EdgeType.CONTAINS)
        contained = list(map(self.codegraph.get_node_by_id, contained_ids))
        return list(filter(lambda n: n.get_type() == NodeType.PACKAGE, contained))

    def query_files(self):
        files = self.get_files()
        s = '\n'.join(list(map(lambda x: x.name, files)))
        return f'仓库{self.path}包含以下文件:\n{s}'

    def get_files(self):
        files = []
        modules = self.get_packages()
        for module in modules:
            files += module.get_files()
        return files

    def query_classes(self):
        classes = self.get_classes()
        s = '\n'.join(list(map(lambda x: x.name, classes)))
        return f'仓库{self.path}包含了以下类:\n{s}'

    def get_classes(self):
        classes = []
        files = self.get_files()
        for file in files:
            classes += file.get_classes()
        return classes

    def __str__(self):
        return self.path

    def __repr__(self):
        return self.path

    def to_dict(self):
        return {"nodeType": NodeType.REPO.name.capitalize(), "nodeId": self.node_id, "path": self.path}
    
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
        return {"nodeType": NodeType.PACKAGE.name.capitalize(), "nodeId": self.node_id, "name": self.name}
    
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

    def query_methods(self):
        methods = self.get_methods()
        s = '\n'.join(list(map(lambda x: x.signature, methods)))
        return f'文件{self.name}中包含以下方法:\n{s}'

    def get_methods(self):
        methods = []
        for clazz in self.get_classes():
            methods += clazz.get_methods()
        return methods

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
        return {"nodeType": NodeType.FILE.name.capitalize(), "nodeId": self.node_id,
                "name": self.name, "path": self.path, "text": self.text, "clean_text": self.clean_text}
    
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
        return {"nodeType": NodeType.TEXTFILE.name.capitalize(), "nodeId": self.node_id, "name": self.name,
                "text": self.text, "path": self.path}
    
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

    def __init__(self, node_id, name, class_type, modifiers, comment, text, start_loc, end_loc, codegraph, clean_text):
        self.node_id = node_id
        self.name = name
        self.class_type = class_type
        self.modifiers = modifiers
        self.comment = comment
        self.text = text
        self.start_loc = start_loc
        self.end_loc = end_loc
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

    def query_methods(self):
        methods = self.get_methods()
        if len(methods) == 0:
            return f'类{self.name}中没有定义方法'
        a = '\n'.join(list(map(lambda m: m.get_simple_signature(), methods)))
        return f"类{self.name}定义了以下方法:\n{a}"

    def get_methods(self):
        return list(filter(lambda n: n.get_type() == NodeType.METHOD.METHOD,
                           list(map(self.codegraph.get_node_by_id,
                                    self.codegraph.get_out_nodes(self.node_id, EdgeType.CONTAINS)))))

    def query_all_methods(self):
        methods = self.get_all_methods()
        if len(methods) == 0:
            return f'类{self.name}中不包含方法'
        a = '\n'.join(list(map(lambda m: m.signature, methods)))
        return f"类{self.name}包含了以下方法:\n{a}"

    def get_all_methods(self):
        methods = self.get_methods()
        for superclass in self.get_superclass_list():
            methods += superclass.get_all_methods()
        return methods

    def query_fields(self):
        fields = self.get_fields()
        if len(fields) == 0:
            return f'类{self.name}中没有定义字段'
        a = '\n'.join(list(map(lambda f: f'{f.name}:{f.field_type}', fields)))
        return f"类{self.name}定义了以下字段:\n{a}"

    def get_fields(self):
        return list(filter(lambda n: n.get_type() == NodeType.FIELD,
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
        return {"nodeType": NodeType.CLASS.name.capitalize(), "nodeId": self.node_id, "name": self.name,
                "classType": self.class_type, "comment": self.comment, "text": self.text, "startLoc": self.start_loc,
                "endLoc": self.end_loc, "modifiers": self.modifiers, "clean_text": self.clean_text}
        
    def get_content(self):

        if self.clean_text:
            return self.name + self.clean_text
        elif self.text:
            return self.name + self.text # 为了兼容没有经过 clean text 处理的数据
        else:
            return self.name


class Field:
    """
    字段节点
    - name: 字段名
    - field_type: 字段名类型
    """

    def __init__(self, node_id, name, field_type, intializer, modifiers, comment, arguments, start_loc, end_loc,
                 codegraph):
        self.node_id = node_id
        self.name = name
        self.field_type = field_type
        self.intializer = intializer
        self.modifiers = modifiers
        self.comment = comment
        self.arguments = arguments
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.codegraph = codegraph

    @staticmethod
    def get_type():
        return NodeType.FIELD

    def __str__(self):
        return self.name + ":" + self.field_type

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {"nodeType": NodeType.FIELD.name.capitalize(), "nodeId": self.node_id, "name": self.name,
                "fieldType": self.field_type, "intializer": self.intializer, "comment": self.comment,
                "arguments": self.arguments, "startLoc": self.start_loc, "col": self.end_loc, "modifiers": self.modifiers}
    
    def get_content(self):
        
        comment = self.comment if self.comment else ""
        modifiers = self.modifiers if self.modifiers else ""
        field_type = self.field_type if self.field_type else ""
        
        if self.comment:
            return comment + "\n" + modifiers +\
            " " + field_type + " " + self.name
        else:
            return modifiers + " " +\
                field_type + " " + self.name


class Method:
    """
    方法节点
    - signature: 方法签名，格式为<class_name>#<method_name>(<params_type>,...)<return_type>
    - text: 方法文本,包含注释、方法签名和方法体等
    """

    def __init__(self, node_id, signature, modifiers, text, comment, class_name, method_name, method_sig, start_loc,
                 end_loc, codegraph):
        self.node_id = node_id
        self.signature = signature
        self.modifiers = modifiers
        self.text = text
        self.comment = comment
        self.class_name = class_name
        self.method_name = method_name
        self.method_sig = method_sig
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.codegraph = codegraph

    @staticmethod
    def get_type():
        return NodeType.METHOD

    def get_simple_signature(self):
        return self.signature[self.signature.index("#") + 1:]

    def query_containing_file(self):
        return f'方法{self.signature}包含在文件{self.get_containing_file().get_path()}中'

    def get_containing_file(self):
        class_ids = self.codegraph.get_in_nodes(self.node_id, EdgeType.CONTAINS)
        if len(class_ids) == 0:
            return None
        class_node_id = class_ids[0]
        return self.codegraph.get_node_by_id(class_node_id).get_containing_file()

    def query_callees(self):
        callees = self.get_callees()
        s = '\n'.join(list(map(lambda x: x.signature, callees)))
        return f'方法{self.signature}直接调用了以下方法:\n{s}'

    def get_callees(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_out_nodes(self.node_id, EdgeType.CALLS)))

    def get_callee_ids(self):
        return self.codegraph.get_out_nodes(self.node_id, EdgeType.CALLS)

    def query_callers(self):
        callers = self.get_callers()
        s = '\n'.join(list(map(lambda x: x.signature, callers)))
        return f'方法{self.signature}被以下方法直接调用了:\n{s}'

    def get_callers(self):
        return list(map(self.codegraph.get_node_by_id, self.codegraph.get_in_nodes(self.node_id, EdgeType.CALLS)))

    def get_caller_ids(self):
        return self.codegraph.get_in_nodes(self.node_id, EdgeType.CALLS)

    def query_common_callees(self, method1):
        common = self.get_common_callee_ids(method1)
        if len(common) == 0:
            return f"方法{self.signature}和方法{method1.signature}没有调用同一个方法"
        s = '\n'.join(list(map(lambda x: x.signature, common)))
        return f"方法{self.signature}和方法{method1.signature}调用了{len(common)}个共同的方法，分别是:\n{s}"

    def get_common_callee_ids(self, method1):
        return set(self.get_callee_ids()) & set(method1.get_callee_ids())

    def query_common_callers(self, method1):
        common = self.get_common_caller_ids(method1)
        if len(common) == 0:
            return f"方法{self.signature}和方法{method1.signature}没有被同一个方法共同调用"
        s = '\n'.join(list(map(lambda x: x.signature, common)))
        return f"方法{self.signature}和方法{method1.signature}被{len(common)}个方法共同调用，分别是:\n{s}"

    def get_common_caller_ids(self, method1):
        return set(self.get_caller_ids()) & set(method1.get_caller_ids())

    def __str__(self):
        return self.signature

    def __repr__(self):
        return self.signature

    def to_dict(self):
        return {"nodeType": NodeType.METHOD.name.capitalize(), "nodeId": self.node_id, "signature": self.signature,
                "method_name": self.method_name, "method_sig": self.method_sig, "comment": self.comment,
                "text": self.text, "startLoc": self.start_loc, "endLoc": self.end_loc, "modifiers": self.modifiers,
                "className": self.class_name}
    
    def get_content(self):
        if self.comment:
            return self.comment + "\n" + self.text
        else:
            return self.text
        


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
        
        # TODO
        # 兼容不同 json 文件格式
        try:
            node_id = node['nodeId']
        except:
            node_id = node['id']
        
        if node_type.upper() == NodeType.REPO.name:
            codegraph.nodes[node_id] = Repo(node_id, node.get('path'), codegraph)
        elif node_type.upper() == NodeType.PACKAGE.name:
            codegraph.nodes[node_id] = Package(node_id, node['name'], codegraph)
        # Handle old graph format
        elif node_type.upper() == "MODULE":
            codegraph.nodes[node_id] = Package(node_id, node['name'], codegraph)
        elif node_type.upper() == NodeType.FILE.name:
            codegraph.nodes[node_id] = File(node_id, node['name'], node.get('path'), node.get('text'), codegraph, node.get('clean_text'))
        elif node_type.upper() == NodeType.TEXTFILE.name:
            codegraph.nodes[node_id] = TextFile(node_id, node['name'], node['text'], node.get('path'), codegraph)
        elif node_type.upper() == NodeType.CLASS.name:

            # Patch：由于 Lib 不连通，考虑不读入 Lib 节点
            ########################################
            if node.get('classType') == 'Lib':
                continue
            ########################################

            codegraph.nodes[node_id] = Class(node_id, node['name'], node.get('classType'), node.get('modifiers'),
                                             node.get('comment'), node.get('text'), node.get('startLoc'),
                                             node.get('endLoc'), codegraph, node.get('clean_text'))
        elif node_type.upper() == NodeType.FIELD.name:
            codegraph.nodes[node_id] = Field(node_id, node['name'], node['fieldType'], node.get('initializer'),
                                             node.get('modifiers'), node.get('comment'), node.get('arguments'),
                                             node.get('startLoc'), node.get('endLoc'), codegraph)
        elif node_type.upper() == NodeType.METHOD.name:
            codegraph.nodes[node_id] = Method(node_id, node['signature'], node['modifiers'], node['text'],
                                              node.get('comment'), node.get('className'), node.get('methodName'),
                                              node.get('methodSig'), node.get('startLoc'), node.get('endLoc'),
                                              codegraph)
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


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
