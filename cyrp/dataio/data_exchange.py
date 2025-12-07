"""
Data Import/Export Module for CYRP
穿黄工程数据导入导出模块

实现多格式数据导入导出、数据转换、批量处理等功能
"""

import asyncio
import csv
import json
import uuid
import io
import os
import zipfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, Type, TypeVar, Generic,
    Iterator, Union, IO
)
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================
# 枚举定义
# ============================================================

class DataFormat(Enum):
    """数据格式"""
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "excel"
    PARQUET = "parquet"
    SQL = "sql"


class ImportStatus(Enum):
    """导入状态"""
    PENDING = "pending"
    VALIDATING = "validating"
    IMPORTING = "importing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ExportStatus(Enum):
    """导出状态"""
    PENDING = "pending"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationLevel(Enum):
    """验证级别"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class FieldMapping:
    """字段映射"""
    source_field: str
    target_field: str
    transform: Optional[Callable] = None
    required: bool = False
    default_value: Any = None
    validator: Optional[Callable] = None


@dataclass
class DataSchema:
    """数据模式"""
    schema_id: str
    name: str
    fields: List[FieldMapping]
    description: str = ""
    version: str = "1.0"

    def get_field(self, source_field: str) -> Optional[FieldMapping]:
        """获取字段映射"""
        for field in self.fields:
            if field.source_field == source_field:
                return field
        return None


@dataclass
class ImportJob:
    """导入任务"""
    job_id: str
    source_file: str
    data_format: DataFormat
    schema_id: str
    status: ImportStatus = ImportStatus.PENDING
    total_records: int = 0
    processed_records: int = 0
    success_records: int = 0
    failed_records: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportJob:
    """导出任务"""
    job_id: str
    target_file: str
    data_format: DataFormat
    schema_id: Optional[str] = None
    status: ExportStatus = ExportStatus.PENDING
    total_records: int = 0
    exported_records: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    record_count: int = 0


@dataclass
class TransformRule:
    """转换规则"""
    rule_id: str
    name: str
    source_type: str
    target_type: str
    transform_func: Callable
    description: str = ""


# ============================================================
# 数据读取器
# ============================================================

class DataReader(ABC):
    """数据读取器基类"""

    @abstractmethod
    async def read(
        self,
        source: Union[str, IO],
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """读取数据"""
        pass

    @abstractmethod
    async def count(self, source: Union[str, IO]) -> int:
        """获取记录数"""
        pass


class CSVReader(DataReader):
    """CSV读取器"""

    async def read(
        self,
        source: Union[str, IO],
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """读取CSV"""
        options = options or {}
        encoding = options.get('encoding', 'utf-8')
        delimiter = options.get('delimiter', ',')
        has_header = options.get('has_header', True)

        if isinstance(source, str):
            with open(source, 'r', encoding=encoding) as f:
                yield from self._read_file(f, delimiter, has_header)
        else:
            yield from self._read_file(source, delimiter, has_header)

    def _read_file(
        self,
        f: IO,
        delimiter: str,
        has_header: bool
    ) -> Iterator[Dict[str, Any]]:
        """读取文件"""
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            headers = next(reader, None)
            if not headers:
                return
            for row in reader:
                yield dict(zip(headers, row))
        else:
            for i, row in enumerate(reader):
                yield {f'col_{j}': v for j, v in enumerate(row)}

    async def count(self, source: Union[str, IO]) -> int:
        """获取记录数"""
        count = 0
        async for _ in self.read(source):
            count += 1
        return count


class JSONReader(DataReader):
    """JSON读取器"""

    async def read(
        self,
        source: Union[str, IO],
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """读取JSON"""
        options = options or {}
        encoding = options.get('encoding', 'utf-8')
        data_path = options.get('data_path', None)

        if isinstance(source, str):
            with open(source, 'r', encoding=encoding) as f:
                data = json.load(f)
        else:
            data = json.load(source)

        # 支持嵌套路径
        if data_path:
            for key in data_path.split('.'):
                data = data.get(key, [])

        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            yield data

    async def count(self, source: Union[str, IO]) -> int:
        """获取记录数"""
        count = 0
        async for _ in self.read(source):
            count += 1
        return count


class XMLReader(DataReader):
    """XML读取器"""

    async def read(
        self,
        source: Union[str, IO],
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """读取XML"""
        options = options or {}
        record_tag = options.get('record_tag', 'record')

        if isinstance(source, str):
            tree = ET.parse(source)
        else:
            tree = ET.parse(source)

        root = tree.getroot()

        for element in root.findall(f'.//{record_tag}'):
            record = {}
            for child in element:
                record[child.tag] = child.text
            yield record

    async def count(self, source: Union[str, IO]) -> int:
        """获取记录数"""
        count = 0
        async for _ in self.read(source):
            count += 1
        return count


# ============================================================
# 数据写入器
# ============================================================

class DataWriter(ABC):
    """数据写入器基类"""

    @abstractmethod
    async def write(
        self,
        records: Iterator[Dict[str, Any]],
        target: Union[str, IO],
        options: Dict[str, Any] = None
    ) -> int:
        """写入数据"""
        pass


class CSVWriter(DataWriter):
    """CSV写入器"""

    async def write(
        self,
        records: Iterator[Dict[str, Any]],
        target: Union[str, IO],
        options: Dict[str, Any] = None
    ) -> int:
        """写入CSV"""
        options = options or {}
        encoding = options.get('encoding', 'utf-8')
        delimiter = options.get('delimiter', ',')
        include_header = options.get('include_header', True)

        count = 0
        headers = None

        if isinstance(target, str):
            with open(target, 'w', encoding=encoding, newline='') as f:
                count = self._write_file(
                    f, records, delimiter, include_header
                )
        else:
            count = self._write_file(
                target, records, delimiter, include_header
            )

        return count

    def _write_file(
        self,
        f: IO,
        records: Iterator[Dict[str, Any]],
        delimiter: str,
        include_header: bool
    ) -> int:
        """写入文件"""
        writer = None
        count = 0

        for record in records:
            if writer is None:
                fieldnames = list(record.keys())
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames, delimiter=delimiter
                )
                if include_header:
                    writer.writeheader()

            writer.writerow(record)
            count += 1

        return count


class JSONWriter(DataWriter):
    """JSON写入器"""

    async def write(
        self,
        records: Iterator[Dict[str, Any]],
        target: Union[str, IO],
        options: Dict[str, Any] = None
    ) -> int:
        """写入JSON"""
        options = options or {}
        encoding = options.get('encoding', 'utf-8')
        indent = options.get('indent', 2)
        as_array = options.get('as_array', True)

        data = list(records)
        count = len(data)

        if not as_array and count == 1:
            data = data[0]

        if isinstance(target, str):
            with open(target, 'w', encoding=encoding) as f:
                json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
        else:
            json.dump(data, target, ensure_ascii=False, indent=indent, default=str)

        return count


class XMLWriter(DataWriter):
    """XML写入器"""

    async def write(
        self,
        records: Iterator[Dict[str, Any]],
        target: Union[str, IO],
        options: Dict[str, Any] = None
    ) -> int:
        """写入XML"""
        options = options or {}
        root_tag = options.get('root_tag', 'data')
        record_tag = options.get('record_tag', 'record')
        encoding = options.get('encoding', 'utf-8')

        root = ET.Element(root_tag)
        count = 0

        for record in records:
            elem = ET.SubElement(root, record_tag)
            for key, value in record.items():
                child = ET.SubElement(elem, key)
                child.text = str(value) if value is not None else ''
            count += 1

        tree = ET.ElementTree(root)

        if isinstance(target, str):
            tree.write(target, encoding=encoding, xml_declaration=True)
        else:
            tree.write(target, encoding=encoding, xml_declaration=True)

        return count


# ============================================================
# 数据验证器
# ============================================================

class DataValidator:
    """数据验证器"""

    def __init__(self):
        self.validators: Dict[str, Callable] = {}
        self._register_default_validators()

    def _register_default_validators(self):
        """注册默认验证器"""
        self.validators['required'] = lambda v: v is not None and v != ''
        self.validators['string'] = lambda v: isinstance(v, str)
        self.validators['number'] = lambda v: isinstance(v, (int, float))
        self.validators['integer'] = lambda v: isinstance(v, int)
        self.validators['positive'] = lambda v: isinstance(v, (int, float)) and v > 0
        self.validators['email'] = lambda v: isinstance(v, str) and '@' in v
        self.validators['date'] = self._validate_date
        self.validators['datetime'] = self._validate_datetime

    def _validate_date(self, value: Any) -> bool:
        """验证日期"""
        if isinstance(value, date):
            return True
        if isinstance(value, str):
            try:
                datetime.strptime(value, '%Y-%m-%d')
                return True
            except ValueError:
                return False
        return False

    def _validate_datetime(self, value: Any) -> bool:
        """验证日期时间"""
        if isinstance(value, datetime):
            return True
        if isinstance(value, str):
            formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']
            for fmt in formats:
                try:
                    datetime.strptime(value, fmt)
                    return True
                except ValueError:
                    continue
        return False

    def register_validator(self, name: str, func: Callable):
        """注册验证器"""
        self.validators[name] = func

    async def validate(
        self,
        records: List[Dict[str, Any]],
        schema: DataSchema,
        level: ValidationLevel = ValidationLevel.BASIC
    ) -> ValidationResult:
        """验证数据"""
        result = ValidationResult(is_valid=True, record_count=len(records))

        if level == ValidationLevel.NONE:
            return result

        for i, record in enumerate(records):
            row_errors = self._validate_record(record, schema, i, level)
            if row_errors:
                result.errors.extend(row_errors)

        result.is_valid = len(result.errors) == 0
        return result

    def _validate_record(
        self,
        record: Dict[str, Any],
        schema: DataSchema,
        row_index: int,
        level: ValidationLevel
    ) -> List[Dict[str, Any]]:
        """验证单条记录"""
        errors = []

        for field_mapping in schema.fields:
            value = record.get(field_mapping.source_field)

            # 必填检查
            if field_mapping.required:
                if not self.validators['required'](value):
                    errors.append({
                        'row': row_index,
                        'field': field_mapping.source_field,
                        'error': 'required field is missing or empty',
                        'value': value
                    })
                    continue

            # 自定义验证器
            if field_mapping.validator and value is not None:
                try:
                    if not field_mapping.validator(value):
                        errors.append({
                            'row': row_index,
                            'field': field_mapping.source_field,
                            'error': 'validation failed',
                            'value': value
                        })
                except Exception as e:
                    errors.append({
                        'row': row_index,
                        'field': field_mapping.source_field,
                        'error': str(e),
                        'value': value
                    })

        return errors


# ============================================================
# 数据转换器
# ============================================================

class DataTransformer:
    """数据转换器"""

    def __init__(self):
        self.transforms: Dict[str, Callable] = {}
        self._register_default_transforms()

    def _register_default_transforms(self):
        """注册默认转换器"""
        self.transforms['to_string'] = str
        self.transforms['to_int'] = lambda v: int(v) if v else 0
        self.transforms['to_float'] = lambda v: float(v) if v else 0.0
        self.transforms['to_bool'] = lambda v: str(v).lower() in ('true', '1', 'yes')
        self.transforms['trim'] = lambda v: str(v).strip() if v else ''
        self.transforms['upper'] = lambda v: str(v).upper() if v else ''
        self.transforms['lower'] = lambda v: str(v).lower() if v else ''
        self.transforms['to_date'] = self._to_date
        self.transforms['to_datetime'] = self._to_datetime

    def _to_date(self, value: Any) -> Optional[date]:
        """转换为日期"""
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return datetime.strptime(value, '%Y-%m-%d').date()
            except ValueError:
                return None
        return None

    def _to_datetime(self, value: Any) -> Optional[datetime]:
        """转换为日期时间"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return None

    def register_transform(self, name: str, func: Callable):
        """注册转换器"""
        self.transforms[name] = func

    async def transform(
        self,
        record: Dict[str, Any],
        schema: DataSchema
    ) -> Dict[str, Any]:
        """转换记录"""
        result = {}

        for field_mapping in schema.fields:
            value = record.get(field_mapping.source_field)

            # 应用默认值
            if value is None and field_mapping.default_value is not None:
                value = field_mapping.default_value

            # 应用转换
            if field_mapping.transform and value is not None:
                try:
                    value = field_mapping.transform(value)
                except Exception as e:
                    logger.warning(
                        f"Transform failed for {field_mapping.source_field}: {e}"
                    )

            result[field_mapping.target_field] = value

        return result

    async def transform_batch(
        self,
        records: List[Dict[str, Any]],
        schema: DataSchema
    ) -> List[Dict[str, Any]]:
        """批量转换"""
        return [await self.transform(r, schema) for r in records]


# ============================================================
# 导入导出服务
# ============================================================

class DataExchangeService:
    """数据导入导出服务"""

    def __init__(self):
        self.readers: Dict[DataFormat, DataReader] = {
            DataFormat.CSV: CSVReader(),
            DataFormat.JSON: JSONReader(),
            DataFormat.XML: XMLReader(),
        }
        self.writers: Dict[DataFormat, DataWriter] = {
            DataFormat.CSV: CSVWriter(),
            DataFormat.JSON: JSONWriter(),
            DataFormat.XML: XMLWriter(),
        }
        self.schemas: Dict[str, DataSchema] = {}
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.jobs: Dict[str, Union[ImportJob, ExportJob]] = {}
        self._lock = asyncio.Lock()

    def register_schema(self, schema: DataSchema) -> str:
        """注册数据模式"""
        self.schemas[schema.schema_id] = schema
        return schema.schema_id

    def get_schema(self, schema_id: str) -> Optional[DataSchema]:
        """获取数据模式"""
        return self.schemas.get(schema_id)

    async def import_data(
        self,
        source: str,
        data_format: DataFormat,
        schema_id: str,
        validation_level: ValidationLevel = ValidationLevel.BASIC,
        options: Dict[str, Any] = None
    ) -> ImportJob:
        """导入数据"""
        job = ImportJob(
            job_id=str(uuid.uuid4()),
            source_file=source,
            data_format=data_format,
            schema_id=schema_id,
            options=options or {}
        )

        async with self._lock:
            self.jobs[job.job_id] = job

        job.started_at = datetime.now()
        job.status = ImportStatus.VALIDATING

        try:
            schema = self.get_schema(schema_id)
            if not schema:
                raise ValueError(f"Schema not found: {schema_id}")

            reader = self.readers.get(data_format)
            if not reader:
                raise ValueError(f"Unsupported format: {data_format}")

            # 读取数据
            records = []
            async for record in reader.read(source, options):
                records.append(record)

            job.total_records = len(records)

            # 验证数据
            validation = await self.validator.validate(
                records, schema, validation_level
            )

            if not validation.is_valid:
                job.errors = validation.errors
                if validation_level == ValidationLevel.STRICT:
                    job.status = ImportStatus.FAILED
                    job.completed_at = datetime.now()
                    return job

            # 转换数据
            job.status = ImportStatus.IMPORTING
            transformed = await self.transformer.transform_batch(records, schema)

            # 处理结果
            job.success_records = len(transformed) - len(validation.errors)
            job.failed_records = len(validation.errors)
            job.processed_records = len(transformed)

            if job.failed_records > 0:
                job.status = ImportStatus.PARTIAL
            else:
                job.status = ImportStatus.COMPLETED

            job.completed_at = datetime.now()

            # 存储转换后的数据（这里可以扩展为写入数据库）
            job.options['transformed_data'] = transformed

        except Exception as e:
            logger.error(f"Import failed: {e}")
            job.status = ImportStatus.FAILED
            job.errors.append({'error': str(e)})
            job.completed_at = datetime.now()

        return job

    async def export_data(
        self,
        records: List[Dict[str, Any]],
        target: str,
        data_format: DataFormat,
        schema_id: Optional[str] = None,
        options: Dict[str, Any] = None
    ) -> ExportJob:
        """导出数据"""
        job = ExportJob(
            job_id=str(uuid.uuid4()),
            target_file=target,
            data_format=data_format,
            schema_id=schema_id,
            total_records=len(records),
            options=options or {}
        )

        async with self._lock:
            self.jobs[job.job_id] = job

        job.started_at = datetime.now()
        job.status = ExportStatus.EXPORTING

        try:
            writer = self.writers.get(data_format)
            if not writer:
                raise ValueError(f"Unsupported format: {data_format}")

            # 如果有模式，进行转换
            if schema_id:
                schema = self.get_schema(schema_id)
                if schema:
                    records = await self.transformer.transform_batch(records, schema)

            # 写入数据
            count = await writer.write(iter(records), target, options)
            job.exported_records = count
            job.status = ExportStatus.COMPLETED
            job.completed_at = datetime.now()

        except Exception as e:
            logger.error(f"Export failed: {e}")
            job.status = ExportStatus.FAILED
            job.completed_at = datetime.now()

        return job

    async def export_to_zip(
        self,
        exports: List[Dict[str, Any]],
        target: str
    ) -> str:
        """导出到ZIP包"""
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as zf:
            for export_config in exports:
                records = export_config.get('records', [])
                filename = export_config.get('filename', 'data.json')
                data_format = export_config.get('format', DataFormat.JSON)

                # 写入临时内存
                buffer = io.StringIO()
                writer = self.writers.get(data_format)
                if writer:
                    await writer.write(iter(records), buffer, {})
                    buffer.seek(0)
                    zf.writestr(filename, buffer.getvalue())

        return target

    async def get_job_status(self, job_id: str) -> Optional[Union[ImportJob, ExportJob]]:
        """获取任务状态"""
        return self.jobs.get(job_id)

    async def create_template(
        self,
        schema_id: str,
        data_format: DataFormat,
        target: str
    ) -> bool:
        """创建导入模板"""
        schema = self.get_schema(schema_id)
        if not schema:
            return False

        # 创建示例记录
        sample = {}
        for field in schema.fields:
            sample[field.source_field] = f"示例_{field.source_field}"

        writer = self.writers.get(data_format)
        if writer:
            await writer.write(iter([sample]), target, {'include_header': True})
            return True

        return False


# ============================================================
# 工厂函数
# ============================================================

def create_cyrp_data_exchange() -> DataExchangeService:
    """创建CYRP数据导入导出服务实例

    Returns:
        DataExchangeService: 数据导入导出服务实例
    """
    service = DataExchangeService()

    # 注册默认模式
    default_schemas = [
        DataSchema(
            schema_id="sensor_data",
            name="传感器数据",
            description="传感器采集数据导入格式",
            fields=[
                FieldMapping("timestamp", "timestamp",
                            transform=service.transformer.transforms['to_datetime'],
                            required=True),
                FieldMapping("sensor_id", "sensor_id", required=True),
                FieldMapping("value", "value",
                            transform=service.transformer.transforms['to_float'],
                            required=True),
                FieldMapping("unit", "unit", default_value=""),
                FieldMapping("quality", "quality",
                            transform=service.transformer.transforms['to_int'],
                            default_value=192),
            ]
        ),
        DataSchema(
            schema_id="equipment_list",
            name="设备清单",
            description="设备资产导入格式",
            fields=[
                FieldMapping("equipment_id", "equipment_id", required=True),
                FieldMapping("name", "name", required=True),
                FieldMapping("type", "equipment_type", required=True),
                FieldMapping("location", "location", default_value=""),
                FieldMapping("manufacturer", "manufacturer", default_value=""),
                FieldMapping("model", "model", default_value=""),
                FieldMapping("install_date", "install_date",
                            transform=service.transformer.transforms['to_date']),
            ]
        ),
        DataSchema(
            schema_id="alarm_records",
            name="报警记录",
            description="报警历史数据导入格式",
            fields=[
                FieldMapping("alarm_id", "alarm_id", required=True),
                FieldMapping("timestamp", "timestamp",
                            transform=service.transformer.transforms['to_datetime'],
                            required=True),
                FieldMapping("source", "source", required=True),
                FieldMapping("level", "level", required=True),
                FieldMapping("message", "message", required=True),
                FieldMapping("acknowledged", "acknowledged",
                            transform=service.transformer.transforms['to_bool'],
                            default_value=False),
            ]
        ),
    ]

    for schema in default_schemas:
        service.register_schema(schema)

    return service
