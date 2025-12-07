"""
API Documentation Generator for CYRP
穿黄工程API文档自动生成系统

功能:
- 自动从代码提取API文档
- 支持多种输出格式(HTML/Markdown/OpenAPI)
- 模块依赖关系图
- 代码示例生成
"""

import ast
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


class DocFormat(Enum):
    """文档格式"""
    MARKDOWN = "md"
    HTML = "html"
    OPENAPI = "json"
    RST = "rst"


@dataclass
class ParameterDoc:
    """参数文档"""
    name: str
    type_hint: str
    default: Optional[str] = None
    description: str = ""
    required: bool = True


@dataclass
class ReturnDoc:
    """返回值文档"""
    type_hint: str
    description: str = ""


@dataclass
class ExceptionDoc:
    """异常文档"""
    exception_type: str
    description: str = ""


@dataclass
class FunctionDoc:
    """函数文档"""
    name: str
    module: str
    docstring: str = ""
    parameters: List[ParameterDoc] = field(default_factory=list)
    returns: Optional[ReturnDoc] = None
    exceptions: List[ExceptionDoc] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    examples: List[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0


@dataclass
class ClassDoc:
    """类文档"""
    name: str
    module: str
    docstring: str = ""
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionDoc] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    class_variables: Dict[str, str] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0


@dataclass
class ModuleDoc:
    """模块文档"""
    name: str
    path: str
    docstring: str = ""
    classes: List[ClassDoc] = field(default_factory=list)
    functions: List[FunctionDoc] = field(default_factory=list)
    constants: Dict[str, Any] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class PackageDoc:
    """包文档"""
    name: str
    path: str
    docstring: str = ""
    modules: List[ModuleDoc] = field(default_factory=list)
    subpackages: List['PackageDoc'] = field(default_factory=list)


class DocstringParser:
    """文档字符串解析器"""

    @staticmethod
    def parse(docstring: str) -> Dict[str, Any]:
        """解析文档字符串"""
        if not docstring:
            return {
                "description": "",
                "parameters": [],
                "returns": None,
                "raises": [],
                "examples": []
            }

        lines = docstring.strip().split('\n')
        result = {
            "description": "",
            "parameters": [],
            "returns": None,
            "raises": [],
            "examples": []
        }

        current_section = "description"
        current_content = []
        current_param = None

        for line in lines:
            stripped = line.strip()

            # 检测章节标记
            if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
                if current_content and current_section == "description":
                    result["description"] = '\n'.join(current_content).strip()
                current_section = "parameters"
                current_content = []
                continue
            elif stripped.lower() in ("returns:", "return:"):
                current_section = "returns"
                current_content = []
                continue
            elif stripped.lower() in ("raises:", "raise:", "exceptions:", "throws:"):
                current_section = "raises"
                current_content = []
                continue
            elif stripped.lower() in ("examples:", "example:", "usage:"):
                current_section = "examples"
                current_content = []
                continue

            # 解析参数
            if current_section == "parameters":
                param_match = re.match(r'(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)', stripped)
                if param_match:
                    if current_param:
                        result["parameters"].append(current_param)
                    current_param = {
                        "name": param_match.group(1),
                        "type": param_match.group(2) or "",
                        "description": param_match.group(3)
                    }
                elif current_param and stripped:
                    current_param["description"] += " " + stripped
            elif current_section == "returns":
                if stripped:
                    if result["returns"] is None:
                        # 检查是否有类型注释
                        type_match = re.match(r'(\w+(?:\[.*?\])?)\s*:\s*(.*)', stripped)
                        if type_match:
                            result["returns"] = {
                                "type": type_match.group(1),
                                "description": type_match.group(2)
                            }
                        else:
                            result["returns"] = {
                                "type": "",
                                "description": stripped
                            }
                    else:
                        result["returns"]["description"] += " " + stripped
            elif current_section == "raises":
                exc_match = re.match(r'(\w+)\s*:\s*(.*)', stripped)
                if exc_match:
                    result["raises"].append({
                        "type": exc_match.group(1),
                        "description": exc_match.group(2)
                    })
            elif current_section == "examples":
                current_content.append(line)
            else:
                current_content.append(stripped)

        # 处理最后的内容
        if current_section == "description" and current_content:
            result["description"] = '\n'.join(current_content).strip()
        elif current_section == "parameters" and current_param:
            result["parameters"].append(current_param)
        elif current_section == "examples" and current_content:
            result["examples"] = ['\n'.join(current_content)]

        return result


class SourceAnalyzer:
    """源代码分析器"""

    def __init__(self):
        self.docstring_parser = DocstringParser()

    def analyze_file(self, file_path: str) -> ModuleDoc:
        """分析源文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        module_name = Path(file_path).stem

        module_doc = ModuleDoc(
            name=module_name,
            path=file_path,
            docstring=ast.get_docstring(tree) or ""
        )

        # 分析导入
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_doc.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_doc.imports.append(node.module)

        # 分析顶层节点
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_doc = self._analyze_class(node, module_name, file_path)
                module_doc.classes.append(class_doc)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_doc = self._analyze_function(node, module_name, file_path)
                module_doc.functions.append(func_doc)
            elif isinstance(node, ast.Assign):
                # 常量
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        module_doc.constants[target.id] = ast.unparse(node.value)

        return module_doc

    def _analyze_class(self, node: ast.ClassDef, module: str, file_path: str) -> ClassDoc:
        """分析类定义"""
        class_doc = ClassDoc(
            name=node.name,
            module=module,
            docstring=ast.get_docstring(node) or "",
            bases=[ast.unparse(base) for base in node.bases],
            decorators=[ast.unparse(d) for d in node.decorator_list],
            source_file=file_path,
            line_number=node.lineno
        )

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_doc = self._analyze_function(item, module, file_path)
                method_doc.is_method = True

                # 检查装饰器
                for decorator in item.decorator_list:
                    dec_name = ast.unparse(decorator)
                    if "classmethod" in dec_name:
                        method_doc.is_classmethod = True
                    elif "staticmethod" in dec_name:
                        method_doc.is_staticmethod = True

                class_doc.methods.append(method_doc)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    type_hint = ast.unparse(item.annotation) if item.annotation else "Any"
                    class_doc.attributes[item.target.id] = type_hint

        return class_doc

    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                          module: str, file_path: str) -> FunctionDoc:
        """分析函数定义"""
        is_async = isinstance(node, ast.AsyncFunctionDef)

        func_doc = FunctionDoc(
            name=node.name,
            module=module,
            docstring=ast.get_docstring(node) or "",
            decorators=[ast.unparse(d) for d in node.decorator_list],
            is_async=is_async,
            source_file=file_path,
            line_number=node.lineno
        )

        # 解析参数
        args = node.args
        defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)

        for arg, default in zip(args.args, defaults):
            type_hint = ast.unparse(arg.annotation) if arg.annotation else "Any"
            param = ParameterDoc(
                name=arg.arg,
                type_hint=type_hint,
                default=ast.unparse(default) if default else None,
                required=default is None
            )
            func_doc.parameters.append(param)

        # 解析返回类型
        if node.returns:
            func_doc.returns = ReturnDoc(
                type_hint=ast.unparse(node.returns)
            )

        # 从文档字符串解析额外信息
        if func_doc.docstring:
            parsed = self.docstring_parser.parse(func_doc.docstring)

            # 更新参数描述
            param_descriptions = {p["name"]: p["description"] for p in parsed["parameters"]}
            for param in func_doc.parameters:
                if param.name in param_descriptions:
                    param.description = param_descriptions[param.name]

            # 更新返回值描述
            if parsed["returns"] and func_doc.returns:
                func_doc.returns.description = parsed["returns"]["description"]

            # 添加异常
            for exc in parsed["raises"]:
                func_doc.exceptions.append(ExceptionDoc(
                    exception_type=exc["type"],
                    description=exc["description"]
                ))

            # 添加示例
            func_doc.examples = parsed["examples"]

        return func_doc

    def analyze_package(self, package_path: str) -> PackageDoc:
        """分析包"""
        package_path = Path(package_path)
        package_name = package_path.name

        package_doc = PackageDoc(
            name=package_name,
            path=str(package_path)
        )

        # 读取__init__.py的文档字符串
        init_file = package_path / "__init__.py"
        if init_file.exists():
            with open(init_file, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)
            package_doc.docstring = ast.get_docstring(tree) or ""

        # 分析所有Python文件
        for py_file in package_path.glob("*.py"):
            if py_file.name != "__init__.py":
                try:
                    module_doc = self.analyze_file(str(py_file))
                    package_doc.modules.append(module_doc)
                except Exception as e:
                    print(f"分析文件失败 {py_file}: {e}")

        # 分析子包
        for subdir in package_path.iterdir():
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                subpackage_doc = self.analyze_package(str(subdir))
                package_doc.subpackages.append(subpackage_doc)

        return package_doc


class MarkdownGenerator:
    """Markdown文档生成器"""

    def generate_module_doc(self, module_doc: ModuleDoc) -> str:
        """生成模块文档"""
        lines = []

        # 标题
        lines.append(f"# {module_doc.name}")
        lines.append("")

        # 模块描述
        if module_doc.docstring:
            lines.append(module_doc.docstring)
            lines.append("")

        # 类列表
        if module_doc.classes:
            lines.append("## Classes")
            lines.append("")
            for class_doc in module_doc.classes:
                lines.append(f"- [{class_doc.name}](#{class_doc.name.lower()})")
            lines.append("")

            # 类详情
            for class_doc in module_doc.classes:
                lines.extend(self._generate_class_section(class_doc))

        # 函数列表
        if module_doc.functions:
            lines.append("## Functions")
            lines.append("")
            for func_doc in module_doc.functions:
                lines.extend(self._generate_function_section(func_doc))

        # 常量
        if module_doc.constants:
            lines.append("## Constants")
            lines.append("")
            lines.append("| Name | Value |")
            lines.append("|------|-------|")
            for name, value in module_doc.constants.items():
                lines.append(f"| `{name}` | `{value}` |")
            lines.append("")

        return '\n'.join(lines)

    def _generate_class_section(self, class_doc: ClassDoc) -> List[str]:
        """生成类章节"""
        lines = []

        # 类标题
        lines.append(f"### {class_doc.name}")
        lines.append("")

        # 继承
        if class_doc.bases:
            lines.append(f"**Inherits from:** {', '.join(class_doc.bases)}")
            lines.append("")

        # 类描述
        if class_doc.docstring:
            lines.append(class_doc.docstring.split('\n')[0])
            lines.append("")

        # 属性
        if class_doc.attributes:
            lines.append("#### Attributes")
            lines.append("")
            lines.append("| Name | Type |")
            lines.append("|------|------|")
            for name, type_hint in class_doc.attributes.items():
                lines.append(f"| `{name}` | `{type_hint}` |")
            lines.append("")

        # 方法
        if class_doc.methods:
            lines.append("#### Methods")
            lines.append("")
            for method in class_doc.methods:
                if not method.name.startswith('_') or method.name in ('__init__', '__call__'):
                    lines.extend(self._generate_method_signature(method))
            lines.append("")

        return lines

    def _generate_function_section(self, func_doc: FunctionDoc) -> List[str]:
        """生成函数章节"""
        lines = []

        # 函数签名
        async_prefix = "async " if func_doc.is_async else ""
        params = ', '.join(
            f"{p.name}: {p.type_hint}" + (f" = {p.default}" if p.default else "")
            for p in func_doc.parameters
        )
        returns = f" -> {func_doc.returns.type_hint}" if func_doc.returns else ""

        lines.append(f"#### `{async_prefix}def {func_doc.name}({params}){returns}`")
        lines.append("")

        # 描述
        if func_doc.docstring:
            desc = func_doc.docstring.split('\n')[0]
            lines.append(desc)
            lines.append("")

        # 参数表
        if func_doc.parameters:
            lines.append("**Parameters:**")
            lines.append("")
            for param in func_doc.parameters:
                req = "(required)" if param.required else "(optional)"
                lines.append(f"- `{param.name}` ({param.type_hint}) {req}: {param.description}")
            lines.append("")

        # 返回值
        if func_doc.returns:
            lines.append(f"**Returns:** `{func_doc.returns.type_hint}` - {func_doc.returns.description}")
            lines.append("")

        # 异常
        if func_doc.exceptions:
            lines.append("**Raises:**")
            lines.append("")
            for exc in func_doc.exceptions:
                lines.append(f"- `{exc.exception_type}`: {exc.description}")
            lines.append("")

        return lines

    def _generate_method_signature(self, method: FunctionDoc) -> List[str]:
        """生成方法签名"""
        lines = []

        prefix = ""
        if method.is_classmethod:
            prefix = "@classmethod "
        elif method.is_staticmethod:
            prefix = "@staticmethod "

        async_prefix = "async " if method.is_async else ""

        params = ', '.join(
            f"{p.name}: {p.type_hint}"
            for p in method.parameters
            if p.name != 'self' and p.name != 'cls'
        )

        returns = f" -> {method.returns.type_hint}" if method.returns else ""

        lines.append(f"- `{prefix}{async_prefix}{method.name}({params}){returns}`")

        if method.docstring:
            desc = method.docstring.split('\n')[0]
            if len(desc) > 100:
                desc = desc[:97] + "..."
            lines.append(f"  - {desc}")

        return lines

    def generate_package_doc(self, package_doc: PackageDoc) -> Dict[str, str]:
        """生成包文档(返回文件名到内容的映射)"""
        docs = {}

        # 生成索引
        index_lines = []
        index_lines.append(f"# {package_doc.name}")
        index_lines.append("")
        if package_doc.docstring:
            index_lines.append(package_doc.docstring)
            index_lines.append("")

        # 模块列表
        if package_doc.modules:
            index_lines.append("## Modules")
            index_lines.append("")
            for module in package_doc.modules:
                desc = module.docstring.split('\n')[0] if module.docstring else ""
                index_lines.append(f"- [{module.name}](./{module.name}.md) - {desc}")
            index_lines.append("")

        # 子包列表
        if package_doc.subpackages:
            index_lines.append("## Subpackages")
            index_lines.append("")
            for subpkg in package_doc.subpackages:
                desc = subpkg.docstring.split('\n')[0] if subpkg.docstring else ""
                index_lines.append(f"- [{subpkg.name}](./{subpkg.name}/README.md) - {desc}")
            index_lines.append("")

        docs["README.md"] = '\n'.join(index_lines)

        # 生成模块文档
        for module in package_doc.modules:
            docs[f"{module.name}.md"] = self.generate_module_doc(module)

        # 递归生成子包文档
        for subpkg in package_doc.subpackages:
            subpkg_docs = self.generate_package_doc(subpkg)
            for filename, content in subpkg_docs.items():
                docs[f"{subpkg.name}/{filename}"] = content

        return docs


class HTMLGenerator:
    """HTML文档生成器"""

    def __init__(self):
        self.md_generator = MarkdownGenerator()

    def generate_module_doc(self, module_doc: ModuleDoc) -> str:
        """生成模块HTML文档"""
        md_content = self.md_generator.generate_module_doc(module_doc)
        return self._md_to_html(md_content, module_doc.name)

    def _md_to_html(self, md_content: str, title: str) -> str:
        """简单的Markdown到HTML转换"""
        html_lines = []

        # HTML头
        html_lines.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - CYRP API Documentation</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        h3 {{ color: #555; }}
        h4 {{ color: #666; font-family: monospace; background: #f5f5f5; padding: 8px; border-radius: 4px; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; }}
        pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #f8f9fa; }}
        .nav {{ background: #333; padding: 10px 20px; margin: -20px -20px 20px -20px; }}
        .nav a {{ color: white; text-decoration: none; margin-right: 20px; }}
        .nav a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <nav class="nav">
        <a href="index.html">首页</a>
        <a href="modules.html">模块</a>
        <a href="classes.html">类</a>
        <a href="functions.html">函数</a>
    </nav>
""")

        # 转换Markdown到HTML(简化版)
        content = md_content
        # 标题
        content = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', content, flags=re.MULTILINE)
        content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
        content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
        content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
        # 粗体
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        # 代码
        content = re.sub(r'`(.+?)`', r'<code>\1</code>', content)
        # 列表
        content = re.sub(r'^- (.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
        # 段落
        content = re.sub(r'\n\n', '</p><p>', content)

        html_lines.append(f"<p>{content}</p>")
        html_lines.append("</body></html>")

        return '\n'.join(html_lines)


class OpenAPIGenerator:
    """OpenAPI文档生成器"""

    def generate(self, package_doc: PackageDoc) -> Dict[str, Any]:
        """生成OpenAPI规范文档"""
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": f"{package_doc.name} API",
                "description": package_doc.docstring,
                "version": "1.0.0"
            },
            "paths": {},
            "components": {
                "schemas": {}
            }
        }

        # 从类生成schema
        for module in package_doc.modules:
            for class_doc in module.classes:
                schema = self._class_to_schema(class_doc)
                spec["components"]["schemas"][class_doc.name] = schema

        return spec

    def _class_to_schema(self, class_doc: ClassDoc) -> Dict[str, Any]:
        """将类转换为OpenAPI schema"""
        schema = {
            "type": "object",
            "description": class_doc.docstring.split('\n')[0] if class_doc.docstring else "",
            "properties": {}
        }

        for attr_name, attr_type in class_doc.attributes.items():
            schema["properties"][attr_name] = self._type_to_schema(attr_type)

        return schema

    def _type_to_schema(self, type_hint: str) -> Dict[str, Any]:
        """将Python类型转换为JSON Schema类型"""
        type_mapping = {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "list": {"type": "array", "items": {}},
            "dict": {"type": "object"},
            "datetime": {"type": "string", "format": "date-time"},
            "Any": {},
        }

        for py_type, json_schema in type_mapping.items():
            if py_type in type_hint:
                return json_schema

        return {"type": "object"}


class APIDocGenerator:
    """API文档生成器主类"""

    def __init__(self, output_dir: str = "./docs/api"):
        self.output_dir = Path(output_dir)
        self.analyzer = SourceAnalyzer()
        self.md_generator = MarkdownGenerator()
        self.html_generator = HTMLGenerator()
        self.openapi_generator = OpenAPIGenerator()

    def generate(
        self,
        source_path: str,
        formats: List[DocFormat] = None
    ) -> Dict[str, str]:
        """生成文档"""
        formats = formats or [DocFormat.MARKDOWN]
        source_path = Path(source_path)
        generated_files = {}

        # 分析源代码
        if source_path.is_dir():
            if (source_path / "__init__.py").exists():
                package_doc = self.analyzer.analyze_package(str(source_path))
                generated_files.update(
                    self._generate_package_docs(package_doc, formats)
                )
            else:
                # 分析目录中的所有Python文件
                for py_file in source_path.glob("**/*.py"):
                    module_doc = self.analyzer.analyze_file(str(py_file))
                    generated_files.update(
                        self._generate_module_docs(module_doc, formats)
                    )
        else:
            module_doc = self.analyzer.analyze_file(str(source_path))
            generated_files.update(
                self._generate_module_docs(module_doc, formats)
            )

        # 写入文件
        for filename, content in generated_files.items():
            file_path = self.output_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        return generated_files

    def _generate_package_docs(
        self,
        package_doc: PackageDoc,
        formats: List[DocFormat]
    ) -> Dict[str, str]:
        """生成包文档"""
        docs = {}

        if DocFormat.MARKDOWN in formats:
            md_docs = self.md_generator.generate_package_doc(package_doc)
            for filename, content in md_docs.items():
                docs[f"md/{filename}"] = content

        if DocFormat.HTML in formats:
            for module in package_doc.modules:
                html_content = self.html_generator.generate_module_doc(module)
                docs[f"html/{module.name}.html"] = html_content

        if DocFormat.OPENAPI in formats:
            openapi_spec = self.openapi_generator.generate(package_doc)
            docs["openapi/spec.json"] = json.dumps(openapi_spec, indent=2, ensure_ascii=False)

        return docs

    def _generate_module_docs(
        self,
        module_doc: ModuleDoc,
        formats: List[DocFormat]
    ) -> Dict[str, str]:
        """生成模块文档"""
        docs = {}

        if DocFormat.MARKDOWN in formats:
            docs[f"md/{module_doc.name}.md"] = self.md_generator.generate_module_doc(module_doc)

        if DocFormat.HTML in formats:
            docs[f"html/{module_doc.name}.html"] = self.html_generator.generate_module_doc(module_doc)

        return docs


def generate_cyrp_docs(output_dir: str = "./docs/api") -> Dict[str, str]:
    """生成穿黄工程API文档"""
    generator = APIDocGenerator(output_dir)

    # 获取cyrp包路径
    cyrp_path = Path(__file__).parent.parent

    # 生成所有格式的文档
    return generator.generate(
        str(cyrp_path),
        formats=[DocFormat.MARKDOWN, DocFormat.HTML, DocFormat.OPENAPI]
    )
