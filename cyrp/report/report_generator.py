"""
Automated Report Generation System for CYRP
穿黄工程自动化报表生成系统

功能:
- 日报/周报/月报自动生成
- 多种格式输出(HTML/PDF/Excel/Word)
- 报表模板管理
- 定时调度执行
- 报表分发管理
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
import csv
import io


class ReportType(Enum):
    """报表类型"""
    DAILY = auto()         # 日报
    WEEKLY = auto()        # 周报
    MONTHLY = auto()       # 月报
    QUARTERLY = auto()     # 季报
    ANNUAL = auto()        # 年报
    SHIFT = auto()         # 班报
    EVENT = auto()         # 事件报告
    ALARM = auto()         # 报警报告
    MAINTENANCE = auto()   # 维护报告
    CUSTOM = auto()        # 自定义报告


class OutputFormat(Enum):
    """输出格式"""
    HTML = "html"
    PDF = "pdf"
    EXCEL = "xlsx"
    WORD = "docx"
    CSV = "csv"
    JSON = "json"
    MARKDOWN = "md"


class ScheduleFrequency(Enum):
    """调度频率"""
    ONCE = auto()
    HOURLY = auto()
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    CRON = auto()


@dataclass
class ReportSection:
    """报表章节"""
    section_id: str
    title: str
    content_type: str  # text, table, chart, image
    data_query: Optional[str] = None
    template: Optional[str] = None
    order: int = 0
    visible: bool = True
    page_break_before: bool = False


@dataclass
class ReportTemplate:
    """报表模板"""
    template_id: str
    name: str
    description: str
    report_type: ReportType
    sections: List[ReportSection] = field(default_factory=list)
    header_template: str = ""
    footer_template: str = ""
    styles: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReportSchedule:
    """报表调度"""
    schedule_id: str
    template_id: str
    name: str
    frequency: ScheduleFrequency
    cron_expression: Optional[str] = None  # 用于CRON频率
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    enabled: bool = True
    output_formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.HTML])
    recipients: List[str] = field(default_factory=list)
    output_path: str = "./reports"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportInstance:
    """报表实例"""
    instance_id: str
    template_id: str
    report_type: ReportType
    title: str
    start_time: datetime
    end_time: datetime
    generated_at: datetime = field(default_factory=datetime.now)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "generated"
    file_paths: Dict[OutputFormat, str] = field(default_factory=dict)


class DataProvider(ABC):
    """数据提供者基类"""

    @abstractmethod
    async def query(self, query: str, parameters: Dict[str, Any]) -> Any:
        """执行数据查询"""
        pass


class InMemoryDataProvider(DataProvider):
    """内存数据提供者(用于测试)"""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def set_data(self, key: str, data: Any):
        """设置数据"""
        self._data[key] = data

    async def query(self, query: str, parameters: Dict[str, Any]) -> Any:
        """查询数据"""
        return self._data.get(query, {})


class ReportFormatter(ABC):
    """报表格式化器基类"""

    @abstractmethod
    def format(self, report: ReportInstance) -> bytes:
        """格式化报表"""
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """获取文件扩展名"""
        pass


class HTMLFormatter(ReportFormatter):
    """HTML格式化器"""

    def __init__(self, include_charts: bool = True):
        self.include_charts = include_charts

    def format(self, report: ReportInstance) -> bytes:
        """生成HTML报表"""
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .report-container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .report-header {{ text-align: center; border-bottom: 2px solid #1890ff; padding-bottom: 20px; margin-bottom: 30px; }}
        .report-title {{ font-size: 28px; color: #333; margin: 0; }}
        .report-subtitle {{ font-size: 14px; color: #666; margin-top: 10px; }}
        .section {{ margin-bottom: 30px; }}
        .section-title {{ font-size: 18px; color: #1890ff; border-left: 4px solid #1890ff; padding-left: 10px; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #1890ff; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .stat-cards {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat-card {{ flex: 1; min-width: 200px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        .chart-container {{ margin: 20px 0; height: 300px; }}
        .report-footer {{ text-align: center; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 12px; }}
        .alert {{ padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .alert-danger {{ background: #ffebee; border-left: 4px solid #f44336; }}
        .alert-warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
        .alert-success {{ background: #e8f5e9; border-left: 4px solid #4caf50; }}
    </style>
    {'<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>' if self.include_charts else ''}
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1 class="report-title">{report.title}</h1>
            <div class="report-subtitle">
                报告周期: {report.start_time.strftime('%Y-%m-%d %H:%M')} 至 {report.end_time.strftime('%Y-%m-%d %H:%M')}<br>
                生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
"""

        # 渲染各章节
        for section in report.sections:
            html += self._render_section(section)

        html += f"""
        <div class="report-footer">
            <p>穿黄工程智能管控系统 - 自动生成报表</p>
            <p>报表ID: {report.instance_id}</p>
        </div>
    </div>
</body>
</html>
"""
        return html.encode('utf-8')

    def _render_section(self, section: Dict[str, Any]) -> str:
        """渲染章节"""
        html = f"""
        <div class="section">
            <h2 class="section-title">{section.get('title', '')}</h2>
"""
        content_type = section.get('content_type', 'text')
        data = section.get('data', {})

        if content_type == 'text':
            html += f"<p>{data.get('text', '')}</p>"
        elif content_type == 'table':
            html += self._render_table(data)
        elif content_type == 'stats':
            html += self._render_stats(data)
        elif content_type == 'chart':
            html += self._render_chart(section.get('section_id', 'chart'), data)
        elif content_type == 'alerts':
            html += self._render_alerts(data)

        html += "</div>"
        return html

    def _render_table(self, data: Dict[str, Any]) -> str:
        """渲染表格"""
        headers = data.get('headers', [])
        rows = data.get('rows', [])

        html = "<table>"
        if headers:
            html += "<thead><tr>"
            for header in headers:
                html += f"<th>{header}</th>"
            html += "</tr></thead>"

        html += "<tbody>"
        for row in rows:
            html += "<tr>"
            for cell in row:
                html += f"<td>{cell}</td>"
            html += "</tr>"
        html += "</tbody></table>"

        return html

    def _render_stats(self, data: Dict[str, Any]) -> str:
        """渲染统计卡片"""
        cards = data.get('cards', [])
        html = '<div class="stat-cards">'
        for card in cards:
            html += f"""
            <div class="stat-card">
                <div class="stat-value">{card.get('value', '')}</div>
                <div class="stat-label">{card.get('label', '')}</div>
            </div>
"""
        html += '</div>'
        return html

    def _render_chart(self, chart_id: str, data: Dict[str, Any]) -> str:
        """渲染图表"""
        if not self.include_charts:
            return "<p>[图表区域]</p>"

        chart_type = data.get('type', 'line')
        chart_data = json.dumps(data.get('data', {}))

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}"></canvas>
            <script>
                new Chart(document.getElementById('{chart_id}'), {{
                    type: '{chart_type}',
                    data: {chart_data},
                    options: {{ responsive: true, maintainAspectRatio: false }}
                }});
            </script>
        </div>
"""

    def _render_alerts(self, data: Dict[str, Any]) -> str:
        """渲染告警列表"""
        alerts = data.get('alerts', [])
        html = ""
        for alert in alerts:
            level = alert.get('level', 'warning')
            html += f"""
            <div class="alert alert-{level}">
                <strong>{alert.get('title', '')}</strong><br>
                {alert.get('message', '')}
                <small style="float:right">{alert.get('time', '')}</small>
            </div>
"""
        return html

    def get_file_extension(self) -> str:
        return "html"


class CSVFormatter(ReportFormatter):
    """CSV格式化器"""

    def format(self, report: ReportInstance) -> bytes:
        """生成CSV报表"""
        output = io.StringIO()
        writer = csv.writer(output)

        # 写入报表头信息
        writer.writerow(['报表标题', report.title])
        writer.writerow(['开始时间', report.start_time.isoformat()])
        writer.writerow(['结束时间', report.end_time.isoformat()])
        writer.writerow(['生成时间', report.generated_at.isoformat()])
        writer.writerow([])

        # 写入各章节
        for section in report.sections:
            writer.writerow([f"=== {section.get('title', '')} ==="])

            content_type = section.get('content_type', 'text')
            data = section.get('data', {})

            if content_type == 'table':
                headers = data.get('headers', [])
                rows = data.get('rows', [])
                if headers:
                    writer.writerow(headers)
                for row in rows:
                    writer.writerow(row)
            elif content_type == 'stats':
                cards = data.get('cards', [])
                for card in cards:
                    writer.writerow([card.get('label', ''), card.get('value', '')])
            elif content_type == 'text':
                writer.writerow([data.get('text', '')])

            writer.writerow([])

        return output.getvalue().encode('utf-8-sig')

    def get_file_extension(self) -> str:
        return "csv"


class JSONFormatter(ReportFormatter):
    """JSON格式化器"""

    def format(self, report: ReportInstance) -> bytes:
        """生成JSON报表"""
        data = {
            "instance_id": report.instance_id,
            "template_id": report.template_id,
            "report_type": report.report_type.name,
            "title": report.title,
            "start_time": report.start_time.isoformat(),
            "end_time": report.end_time.isoformat(),
            "generated_at": report.generated_at.isoformat(),
            "parameters": report.parameters,
            "sections": report.sections,
            "status": report.status
        }
        return json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')

    def get_file_extension(self) -> str:
        return "json"


class MarkdownFormatter(ReportFormatter):
    """Markdown格式化器"""

    def format(self, report: ReportInstance) -> bytes:
        """生成Markdown报表"""
        md = f"""# {report.title}

**报告周期**: {report.start_time.strftime('%Y-%m-%d %H:%M')} 至 {report.end_time.strftime('%Y-%m-%d %H:%M')}

**生成时间**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

---

"""
        for section in report.sections:
            md += f"## {section.get('title', '')}\n\n"

            content_type = section.get('content_type', 'text')
            data = section.get('data', {})

            if content_type == 'text':
                md += f"{data.get('text', '')}\n\n"
            elif content_type == 'table':
                md += self._render_table(data)
            elif content_type == 'stats':
                md += self._render_stats(data)
            elif content_type == 'alerts':
                md += self._render_alerts(data)

        md += f"""
---

*穿黄工程智能管控系统 - 报表ID: {report.instance_id}*
"""
        return md.encode('utf-8')

    def _render_table(self, data: Dict[str, Any]) -> str:
        """渲染Markdown表格"""
        headers = data.get('headers', [])
        rows = data.get('rows', [])

        if not headers:
            return ""

        md = "| " + " | ".join(str(h) for h in headers) + " |\n"
        md += "| " + " | ".join("---" for _ in headers) + " |\n"

        for row in rows:
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"

        return md + "\n"

    def _render_stats(self, data: Dict[str, Any]) -> str:
        """渲染统计信息"""
        cards = data.get('cards', [])
        md = ""
        for card in cards:
            md += f"- **{card.get('label', '')}**: {card.get('value', '')}\n"
        return md + "\n"

    def _render_alerts(self, data: Dict[str, Any]) -> str:
        """渲染告警"""
        alerts = data.get('alerts', [])
        md = ""
        for alert in alerts:
            level = alert.get('level', 'warning').upper()
            md += f"- [{level}] **{alert.get('title', '')}**: {alert.get('message', '')} ({alert.get('time', '')})\n"
        return md + "\n"

    def get_file_extension(self) -> str:
        return "md"


class ReportBuilder:
    """报表构建器"""

    def __init__(self, template: ReportTemplate, data_provider: DataProvider):
        self.template = template
        self.data_provider = data_provider

    async def build(
        self,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ReportInstance:
        """构建报表"""
        instance_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 合并参数
        merged_params = {**self.template.parameters, **(parameters or {})}
        merged_params['start_time'] = start_time
        merged_params['end_time'] = end_time

        # 生成报表标题
        title = self._generate_title(start_time, end_time)

        # 构建各章节
        sections = []
        for section_config in sorted(self.template.sections, key=lambda s: s.order):
            if not section_config.visible:
                continue

            section_data = await self._build_section(section_config, merged_params)
            sections.append(section_data)

        return ReportInstance(
            instance_id=instance_id,
            template_id=self.template.template_id,
            report_type=self.template.report_type,
            title=title,
            start_time=start_time,
            end_time=end_time,
            sections=sections,
            parameters=merged_params
        )

    def _generate_title(self, start_time: datetime, end_time: datetime) -> str:
        """生成报表标题"""
        report_type = self.template.report_type

        if report_type == ReportType.DAILY:
            return f"穿黄工程日报 - {start_time.strftime('%Y年%m月%d日')}"
        elif report_type == ReportType.WEEKLY:
            return f"穿黄工程周报 - {start_time.strftime('%Y年第%W周')}"
        elif report_type == ReportType.MONTHLY:
            return f"穿黄工程月报 - {start_time.strftime('%Y年%m月')}"
        elif report_type == ReportType.SHIFT:
            return f"穿黄工程班报 - {start_time.strftime('%Y-%m-%d %H:%M')}"
        elif report_type == ReportType.ALARM:
            return f"穿黄工程报警报告 - {start_time.strftime('%Y-%m-%d')}"
        elif report_type == ReportType.MAINTENANCE:
            return f"穿黄工程维护报告 - {start_time.strftime('%Y-%m-%d')}"
        else:
            return f"穿黄工程报告 - {self.template.name}"

    async def _build_section(
        self,
        section_config: ReportSection,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建章节"""
        section = {
            "section_id": section_config.section_id,
            "title": section_config.title,
            "content_type": section_config.content_type,
            "data": {}
        }

        if section_config.data_query:
            # 执行数据查询
            query_result = await self.data_provider.query(
                section_config.data_query, parameters
            )
            section["data"] = query_result

        return section


class ReportDistributor:
    """报表分发器"""

    def __init__(self):
        self._channels: Dict[str, Callable] = {}

    def register_channel(self, channel_name: str, handler: Callable):
        """注册分发渠道"""
        self._channels[channel_name] = handler

    async def distribute(
        self,
        report: ReportInstance,
        file_paths: Dict[OutputFormat, str],
        recipients: List[str],
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """分发报表"""
        results = {}
        target_channels = channels or list(self._channels.keys())

        for channel_name in target_channels:
            if channel_name in self._channels:
                try:
                    handler = self._channels[channel_name]
                    await handler(report, file_paths, recipients)
                    results[channel_name] = True
                except Exception as e:
                    results[channel_name] = False

        return results


class ReportScheduler:
    """报表调度器"""

    def __init__(
        self,
        report_manager: 'ReportManager'
    ):
        self.report_manager = report_manager
        self.schedules: Dict[str, ReportSchedule] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None

    def add_schedule(self, schedule: ReportSchedule):
        """添加调度"""
        # 计算下次运行时间
        schedule.next_run = self._calculate_next_run(schedule)
        self.schedules[schedule.schedule_id] = schedule

    def remove_schedule(self, schedule_id: str):
        """移除调度"""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]

    def enable_schedule(self, schedule_id: str, enabled: bool = True):
        """启用/禁用调度"""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = enabled

    async def start(self):
        """启动调度器"""
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self):
        """调度循环"""
        while self._running:
            now = datetime.now()

            for schedule in list(self.schedules.values()):
                if not schedule.enabled:
                    continue

                if schedule.next_run and schedule.next_run <= now:
                    # 执行报表生成
                    try:
                        await self._execute_schedule(schedule)
                    except Exception:
                        pass

                    # 更新下次运行时间
                    schedule.last_run = now
                    schedule.next_run = self._calculate_next_run(schedule)

            await asyncio.sleep(60)  # 每分钟检查一次

    async def _execute_schedule(self, schedule: ReportSchedule):
        """执行调度任务"""
        # 计算报表时间范围
        end_time = datetime.now()
        start_time = self._calculate_start_time(schedule, end_time)

        # 生成报表
        await self.report_manager.generate_report(
            template_id=schedule.template_id,
            start_time=start_time,
            end_time=end_time,
            output_formats=schedule.output_formats,
            output_path=schedule.output_path,
            parameters=schedule.parameters
        )

    def _calculate_next_run(self, schedule: ReportSchedule) -> datetime:
        """计算下次运行时间"""
        now = datetime.now()

        if schedule.frequency == ScheduleFrequency.ONCE:
            return schedule.next_run or now

        elif schedule.frequency == ScheduleFrequency.HOURLY:
            next_run = now.replace(minute=0, second=0, microsecond=0)
            next_run += timedelta(hours=1)
            return next_run

        elif schedule.frequency == ScheduleFrequency.DAILY:
            next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run

        elif schedule.frequency == ScheduleFrequency.WEEKLY:
            # 每周一早上6点
            days_ahead = 0 - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
            next_run += timedelta(days=days_ahead)
            return next_run

        elif schedule.frequency == ScheduleFrequency.MONTHLY:
            # 每月1号早上6点
            if now.day == 1 and now.hour < 6:
                next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
            else:
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=1,
                                           hour=6, minute=0, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month + 1, day=1,
                                           hour=6, minute=0, second=0, microsecond=0)
            return next_run

        return now + timedelta(days=1)

    def _calculate_start_time(
        self,
        schedule: ReportSchedule,
        end_time: datetime
    ) -> datetime:
        """计算报表开始时间"""
        if schedule.frequency == ScheduleFrequency.HOURLY:
            return end_time - timedelta(hours=1)
        elif schedule.frequency == ScheduleFrequency.DAILY:
            return end_time - timedelta(days=1)
        elif schedule.frequency == ScheduleFrequency.WEEKLY:
            return end_time - timedelta(weeks=1)
        elif schedule.frequency == ScheduleFrequency.MONTHLY:
            return end_time - timedelta(days=30)
        else:
            return end_time - timedelta(days=1)


class ReportManager:
    """报表管理器"""

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.templates: Dict[str, ReportTemplate] = {}
        self.formatters: Dict[OutputFormat, ReportFormatter] = {
            OutputFormat.HTML: HTMLFormatter(),
            OutputFormat.CSV: CSVFormatter(),
            OutputFormat.JSON: JSONFormatter(),
            OutputFormat.MARKDOWN: MarkdownFormatter(),
        }
        self.distributor = ReportDistributor()
        self.scheduler = ReportScheduler(self)
        self._report_history: List[ReportInstance] = []

    def register_template(self, template: ReportTemplate):
        """注册报表模板"""
        self.templates[template.template_id] = template

    def register_formatter(self, format: OutputFormat, formatter: ReportFormatter):
        """注册格式化器"""
        self.formatters[format] = formatter

    async def generate_report(
        self,
        template_id: str,
        start_time: datetime,
        end_time: datetime,
        output_formats: Optional[List[OutputFormat]] = None,
        output_path: str = "./reports",
        parameters: Optional[Dict[str, Any]] = None
    ) -> ReportInstance:
        """生成报表"""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")

        template = self.templates[template_id]
        builder = ReportBuilder(template, self.data_provider)

        # 构建报表
        report = await builder.build(start_time, end_time, parameters)

        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # 生成各种格式的输出
        formats = output_formats or [OutputFormat.HTML]
        for fmt in formats:
            if fmt in self.formatters:
                formatter = self.formatters[fmt]
                content = formatter.format(report)

                # 保存文件
                file_name = f"{report.instance_id}.{formatter.get_file_extension()}"
                file_path = os.path.join(output_path, file_name)

                with open(file_path, 'wb') as f:
                    f.write(content)

                report.file_paths[fmt] = file_path

        self._report_history.append(report)
        return report

    async def get_report_history(
        self,
        template_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ReportInstance]:
        """获取报表历史"""
        reports = self._report_history

        if template_id:
            reports = [r for r in reports if r.template_id == template_id]

        if start_time:
            reports = [r for r in reports if r.generated_at >= start_time]

        if end_time:
            reports = [r for r in reports if r.generated_at <= end_time]

        return reports[-limit:]


def create_cyrp_report_system() -> ReportManager:
    """创建穿黄工程报表系统"""
    data_provider = InMemoryDataProvider()
    manager = ReportManager(data_provider)

    # 日报模板
    daily_template = ReportTemplate(
        template_id="daily_report",
        name="穿黄工程日报",
        description="每日运行情况汇总报告",
        report_type=ReportType.DAILY,
        sections=[
            ReportSection(
                section_id="summary",
                title="运行概况",
                content_type="stats",
                data_query="daily_summary",
                order=1
            ),
            ReportSection(
                section_id="flow_data",
                title="水量统计",
                content_type="table",
                data_query="flow_statistics",
                order=2
            ),
            ReportSection(
                section_id="flow_chart",
                title="流量趋势",
                content_type="chart",
                data_query="flow_trend",
                order=3
            ),
            ReportSection(
                section_id="equipment",
                title="设备运行状态",
                content_type="table",
                data_query="equipment_status",
                order=4
            ),
            ReportSection(
                section_id="alarms",
                title="报警记录",
                content_type="alerts",
                data_query="alarm_records",
                order=5
            ),
            ReportSection(
                section_id="notes",
                title="运行备注",
                content_type="text",
                data_query="operation_notes",
                order=6
            ),
        ]
    )
    manager.register_template(daily_template)

    # 周报模板
    weekly_template = ReportTemplate(
        template_id="weekly_report",
        name="穿黄工程周报",
        description="每周运行分析报告",
        report_type=ReportType.WEEKLY,
        sections=[
            ReportSection(
                section_id="weekly_summary",
                title="本周概况",
                content_type="stats",
                data_query="weekly_summary",
                order=1
            ),
            ReportSection(
                section_id="daily_comparison",
                title="日均数据对比",
                content_type="table",
                data_query="daily_comparison",
                order=2
            ),
            ReportSection(
                section_id="trend_analysis",
                title="趋势分析",
                content_type="chart",
                data_query="weekly_trend",
                order=3
            ),
            ReportSection(
                section_id="maintenance_summary",
                title="维护工作汇总",
                content_type="table",
                data_query="maintenance_summary",
                order=4
            ),
            ReportSection(
                section_id="issues",
                title="问题与建议",
                content_type="text",
                data_query="weekly_issues",
                order=5
            ),
        ]
    )
    manager.register_template(weekly_template)

    # 月报模板
    monthly_template = ReportTemplate(
        template_id="monthly_report",
        name="穿黄工程月报",
        description="月度综合分析报告",
        report_type=ReportType.MONTHLY,
        sections=[
            ReportSection(
                section_id="monthly_summary",
                title="月度概况",
                content_type="stats",
                data_query="monthly_summary",
                order=1
            ),
            ReportSection(
                section_id="water_transfer",
                title="调水完成情况",
                content_type="table",
                data_query="water_transfer_progress",
                order=2
            ),
            ReportSection(
                section_id="equipment_health",
                title="设备健康分析",
                content_type="chart",
                data_query="equipment_health",
                order=3
            ),
            ReportSection(
                section_id="energy_analysis",
                title="能耗分析",
                content_type="table",
                data_query="energy_analysis",
                order=4
            ),
            ReportSection(
                section_id="safety_report",
                title="安全生产情况",
                content_type="text",
                data_query="safety_report",
                order=5
            ),
            ReportSection(
                section_id="next_month_plan",
                title="下月工作计划",
                content_type="text",
                data_query="next_month_plan",
                order=6
            ),
        ]
    )
    manager.register_template(monthly_template)

    # 班报模板
    shift_template = ReportTemplate(
        template_id="shift_report",
        name="穿黄工程班报",
        description="班组交接报告",
        report_type=ReportType.SHIFT,
        sections=[
            ReportSection(
                section_id="shift_summary",
                title="本班概况",
                content_type="stats",
                data_query="shift_summary",
                order=1
            ),
            ReportSection(
                section_id="operations",
                title="操作记录",
                content_type="table",
                data_query="operation_log",
                order=2
            ),
            ReportSection(
                section_id="handover",
                title="交接事项",
                content_type="text",
                data_query="handover_notes",
                order=3
            ),
        ]
    )
    manager.register_template(shift_template)

    # 报警报告模板
    alarm_template = ReportTemplate(
        template_id="alarm_report",
        name="报警分析报告",
        description="报警事件分析报告",
        report_type=ReportType.ALARM,
        sections=[
            ReportSection(
                section_id="alarm_stats",
                title="报警统计",
                content_type="stats",
                data_query="alarm_statistics",
                order=1
            ),
            ReportSection(
                section_id="alarm_distribution",
                title="报警分布",
                content_type="chart",
                data_query="alarm_distribution",
                order=2
            ),
            ReportSection(
                section_id="alarm_details",
                title="报警明细",
                content_type="table",
                data_query="alarm_details",
                order=3
            ),
            ReportSection(
                section_id="alarm_analysis",
                title="原因分析",
                content_type="text",
                data_query="alarm_analysis",
                order=4
            ),
        ]
    )
    manager.register_template(alarm_template)

    # 维护报告模板
    maintenance_template = ReportTemplate(
        template_id="maintenance_report",
        name="设备维护报告",
        description="设备维护保养报告",
        report_type=ReportType.MAINTENANCE,
        sections=[
            ReportSection(
                section_id="maintenance_summary",
                title="维护概况",
                content_type="stats",
                data_query="maintenance_summary",
                order=1
            ),
            ReportSection(
                section_id="completed_tasks",
                title="已完成维护",
                content_type="table",
                data_query="completed_maintenance",
                order=2
            ),
            ReportSection(
                section_id="pending_tasks",
                title="待处理项目",
                content_type="table",
                data_query="pending_maintenance",
                order=3
            ),
            ReportSection(
                section_id="spare_parts",
                title="备件使用情况",
                content_type="table",
                data_query="spare_parts_usage",
                order=4
            ),
        ]
    )
    manager.register_template(maintenance_template)

    return manager
