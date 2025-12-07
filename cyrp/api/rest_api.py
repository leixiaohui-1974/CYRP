"""
REST API Module for CYRP
穿黄工程RESTful API接口模块

功能:
- 完整的REST API端点
- JWT认证
- 请求验证与错误处理
- API版本管理
- 速率限制
- OpenAPI文档自动生成
"""

import asyncio
import functools
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import base64


class HTTPMethod(Enum):
    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"


class ContentType(Enum):
    """内容类型"""
    JSON = "application/json"
    XML = "application/xml"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"


@dataclass
class APIRequest:
    """API请求"""
    request_id: str
    method: HTTPMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    path_params: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    client_ip: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None


@dataclass
class APIResponse:
    """API响应"""
    status_code: int
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    content_type: ContentType = ContentType.JSON

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status_code": self.status_code,
            "headers": self.headers,
            "body": self.body
        }


@dataclass
class APIError:
    """API错误"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    status_code: int = 400

    def to_response(self) -> APIResponse:
        """转换为API响应"""
        return APIResponse(
            status_code=self.status_code,
            body={
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "details": self.details
                }
            }
        )


@dataclass
class RouteDefinition:
    """路由定义"""
    path: str
    method: HTTPMethod
    handler: Callable
    summary: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    auth_required: bool = True
    rate_limit: int = 0  # 每分钟请求数,0表示无限制
    request_schema: Optional[Dict] = None
    response_schema: Optional[Dict] = None


class JWTManager:
    """JWT管理器"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key.encode()
        self.algorithm = algorithm
        self.token_expiry_hours = 24

    def create_token(
        self,
        user_id: str,
        username: str,
        roles: List[str],
        extra_claims: Optional[Dict] = None
    ) -> str:
        """创建JWT令牌"""
        now = datetime.now()
        payload = {
            "sub": user_id,
            "username": username,
            "roles": roles,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=self.token_expiry_hours)).timestamp()),
            "jti": str(uuid.uuid4())
        }
        if extra_claims:
            payload.update(extra_claims)

        # 编码头部
        header = {"alg": self.algorithm, "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).rstrip(b'=').decode()

        # 编码载荷
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b'=').decode()

        # 签名
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret_key,
            message.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b'=').decode()

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict], str]:
        """验证JWT令牌"""
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return False, None, "无效的令牌格式"

            header_b64, payload_b64, signature_b64 = parts

            # 验证签名
            message = f"{header_b64}.{payload_b64}"
            expected_signature = hmac.new(
                self.secret_key,
                message.encode(),
                hashlib.sha256
            ).digest()
            expected_signature_b64 = base64.urlsafe_b64encode(expected_signature).rstrip(b'=').decode()

            if not hmac.compare_digest(signature_b64, expected_signature_b64):
                return False, None, "签名验证失败"

            # 解码载荷
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += '=' * padding
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))

            # 检查过期
            if payload.get("exp", 0) < time.time():
                return False, None, "令牌已过期"

            return True, payload, ""

        except Exception as e:
            return False, None, str(e)

    def refresh_token(self, token: str) -> Optional[str]:
        """刷新令牌"""
        valid, payload, _ = self.verify_token(token)
        if not valid or not payload:
            return None

        return self.create_token(
            payload["sub"],
            payload["username"],
            payload["roles"]
        )


class RateLimiter:
    """速率限制器"""

    def __init__(self):
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """检查是否超过限制"""
        async with self._lock:
            now = time.time()
            cutoff = now - window_seconds

            if key not in self._requests:
                self._requests[key] = []

            # 清理过期记录
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]

            # 检查限制
            if len(self._requests[key]) >= limit:
                return False

            # 记录请求
            self._requests[key].append(now)
            return True

    async def get_remaining(self, key: str, limit: int, window_seconds: int = 60) -> int:
        """获取剩余请求数"""
        async with self._lock:
            now = time.time()
            cutoff = now - window_seconds

            if key not in self._requests:
                return limit

            valid_requests = [t for t in self._requests[key] if t > cutoff]
            return max(0, limit - len(valid_requests))


class RequestValidator:
    """请求验证器"""

    @staticmethod
    def validate(data: Any, schema: Dict) -> Tuple[bool, List[str]]:
        """验证数据是否符合模式"""
        errors = []

        if not isinstance(data, dict):
            return False, ["请求体必须是对象"]

        for field_name, field_schema in schema.items():
            required = field_schema.get("required", False)
            field_type = field_schema.get("type", "string")

            if field_name not in data:
                if required:
                    errors.append(f"缺少必需字段: {field_name}")
                continue

            value = data[field_name]

            # 类型验证
            if field_type == "string" and not isinstance(value, str):
                errors.append(f"字段 {field_name} 必须是字符串")
            elif field_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"字段 {field_name} 必须是数字")
            elif field_type == "integer" and not isinstance(value, int):
                errors.append(f"字段 {field_name} 必须是整数")
            elif field_type == "boolean" and not isinstance(value, bool):
                errors.append(f"字段 {field_name} 必须是布尔值")
            elif field_type == "array" and not isinstance(value, list):
                errors.append(f"字段 {field_name} 必须是数组")
            elif field_type == "object" and not isinstance(value, dict):
                errors.append(f"字段 {field_name} 必须是对象")

            # 范围验证
            if isinstance(value, (int, float)):
                if "minimum" in field_schema and value < field_schema["minimum"]:
                    errors.append(f"字段 {field_name} 不能小于 {field_schema['minimum']}")
                if "maximum" in field_schema and value > field_schema["maximum"]:
                    errors.append(f"字段 {field_name} 不能大于 {field_schema['maximum']}")

            # 长度验证
            if isinstance(value, str):
                if "minLength" in field_schema and len(value) < field_schema["minLength"]:
                    errors.append(f"字段 {field_name} 长度不能小于 {field_schema['minLength']}")
                if "maxLength" in field_schema and len(value) > field_schema["maxLength"]:
                    errors.append(f"字段 {field_name} 长度不能大于 {field_schema['maxLength']}")

            # 枚举验证
            if "enum" in field_schema and value not in field_schema["enum"]:
                errors.append(f"字段 {field_name} 必须是以下值之一: {field_schema['enum']}")

        return len(errors) == 0, errors


class APIRouter:
    """API路由器"""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.routes: List[RouteDefinition] = []
        self._middleware: List[Callable] = []

    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self._middleware.append(middleware)

    def route(
        self,
        path: str,
        method: HTTPMethod = HTTPMethod.GET,
        **kwargs
    ):
        """路由装饰器"""
        def decorator(handler: Callable):
            full_path = f"{self.prefix}{path}"
            route_def = RouteDefinition(
                path=full_path,
                method=method,
                handler=handler,
                **kwargs
            )
            self.routes.append(route_def)
            return handler
        return decorator

    def get(self, path: str, **kwargs):
        """GET路由"""
        return self.route(path, HTTPMethod.GET, **kwargs)

    def post(self, path: str, **kwargs):
        """POST路由"""
        return self.route(path, HTTPMethod.POST, **kwargs)

    def put(self, path: str, **kwargs):
        """PUT路由"""
        return self.route(path, HTTPMethod.PUT, **kwargs)

    def delete(self, path: str, **kwargs):
        """DELETE路由"""
        return self.route(path, HTTPMethod.DELETE, **kwargs)

    def patch(self, path: str, **kwargs):
        """PATCH路由"""
        return self.route(path, HTTPMethod.PATCH, **kwargs)


class APIServer:
    """API服务器"""

    def __init__(
        self,
        jwt_secret: str = "cyrp-api-secret",
        enable_cors: bool = True
    ):
        self.jwt_manager = JWTManager(jwt_secret)
        self.rate_limiter = RateLimiter()
        self.enable_cors = enable_cors
        self._routers: List[APIRouter] = []
        self._routes: Dict[Tuple[str, HTTPMethod], RouteDefinition] = {}
        self._request_log: List[Dict] = []

    def include_router(self, router: APIRouter):
        """包含路由器"""
        self._routers.append(router)
        for route in router.routes:
            self._routes[(route.path, route.method)] = route

    async def handle_request(self, request: APIRequest) -> APIResponse:
        """处理请求"""
        start_time = time.perf_counter()

        try:
            # CORS预检
            if request.method == HTTPMethod.OPTIONS:
                return self._handle_cors_preflight()

            # 查找路由
            route = self._match_route(request)
            if not route:
                return APIError(
                    code="NOT_FOUND",
                    message=f"路径不存在: {request.path}",
                    status_code=404
                ).to_response()

            # 认证检查
            if route.auth_required:
                auth_result = await self._check_auth(request)
                if auth_result:
                    return auth_result

            # 速率限制检查
            if route.rate_limit > 0:
                rate_key = f"{request.client_ip}:{request.path}"
                allowed = await self.rate_limiter.check(rate_key, route.rate_limit)
                if not allowed:
                    return APIError(
                        code="RATE_LIMIT_EXCEEDED",
                        message="请求过于频繁,请稍后再试",
                        status_code=429
                    ).to_response()

            # 请求验证
            if route.request_schema and request.body:
                valid, errors = RequestValidator.validate(request.body, route.request_schema)
                if not valid:
                    return APIError(
                        code="VALIDATION_ERROR",
                        message="请求参数验证失败",
                        details={"errors": errors},
                        status_code=400
                    ).to_response()

            # 执行处理器
            if asyncio.iscoroutinefunction(route.handler):
                result = await route.handler(request)
            else:
                result = route.handler(request)

            # 处理响应
            if isinstance(result, APIResponse):
                response = result
            elif isinstance(result, dict):
                response = APIResponse(status_code=200, body=result)
            else:
                response = APIResponse(status_code=200, body={"data": result})

            # 添加CORS头
            if self.enable_cors:
                response.headers["Access-Control-Allow-Origin"] = "*"

        except Exception as e:
            response = APIError(
                code="INTERNAL_ERROR",
                message=str(e),
                status_code=500
            ).to_response()

        # 记录请求日志
        processing_time = (time.perf_counter() - start_time) * 1000
        self._log_request(request, response, processing_time)

        return response

    def _match_route(self, request: APIRequest) -> Optional[RouteDefinition]:
        """匹配路由"""
        # 精确匹配
        route = self._routes.get((request.path, request.method))
        if route:
            return route

        # 路径参数匹配
        for (path, method), route in self._routes.items():
            if method != request.method:
                continue

            match, params = self._match_path(path, request.path)
            if match:
                request.path_params = params
                return route

        return None

    def _match_path(self, pattern: str, path: str) -> Tuple[bool, Dict[str, str]]:
        """匹配路径模式"""
        pattern_parts = pattern.split('/')
        path_parts = path.split('/')

        if len(pattern_parts) != len(path_parts):
            return False, {}

        params = {}
        for pp, pathp in zip(pattern_parts, path_parts):
            if pp.startswith('{') and pp.endswith('}'):
                param_name = pp[1:-1]
                params[param_name] = pathp
            elif pp != pathp:
                return False, {}

        return True, params

    async def _check_auth(self, request: APIRequest) -> Optional[APIResponse]:
        """检查认证"""
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return APIError(
                code="UNAUTHORIZED",
                message="缺少认证令牌",
                status_code=401
            ).to_response()

        token = auth_header[7:]
        valid, payload, error = self.jwt_manager.verify_token(token)

        if not valid:
            return APIError(
                code="UNAUTHORIZED",
                message=error or "认证失败",
                status_code=401
            ).to_response()

        request.user_id = payload.get("sub")
        return None

    def _handle_cors_preflight(self) -> APIResponse:
        """处理CORS预检请求"""
        return APIResponse(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400"
            }
        )

    def _log_request(
        self,
        request: APIRequest,
        response: APIResponse,
        processing_time: float
    ):
        """记录请求日志"""
        self._request_log.append({
            "request_id": request.request_id,
            "method": request.method.value,
            "path": request.path,
            "status_code": response.status_code,
            "processing_time_ms": processing_time,
            "client_ip": request.client_ip,
            "user_id": request.user_id,
            "timestamp": request.timestamp.isoformat()
        })

        # 保持日志大小
        if len(self._request_log) > 10000:
            self._request_log = self._request_log[-5000:]

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """生成OpenAPI规范"""
        paths = {}

        for (path, method), route in self._routes.items():
            if path not in paths:
                paths[path] = {}

            operation = {
                "summary": route.summary,
                "description": route.description,
                "tags": route.tags,
                "responses": {
                    "200": {"description": "成功"}
                }
            }

            if route.auth_required:
                operation["security"] = [{"bearerAuth": []}]

            if route.request_schema:
                operation["requestBody"] = {
                    "content": {
                        "application/json": {
                            "schema": self._schema_to_openapi(route.request_schema)
                        }
                    }
                }

            paths[path][method.value.lower()] = operation

        return {
            "openapi": "3.0.3",
            "info": {
                "title": "CYRP API",
                "description": "穿黄工程智能管控系统API",
                "version": "1.0.0"
            },
            "servers": [
                {"url": "/api/v1", "description": "API服务器"}
            ],
            "paths": paths,
            "components": {
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            }
        }

    def _schema_to_openapi(self, schema: Dict) -> Dict:
        """转换为OpenAPI模式"""
        properties = {}
        required = []

        for field_name, field_schema in schema.items():
            prop = {"type": field_schema.get("type", "string")}
            if "minimum" in field_schema:
                prop["minimum"] = field_schema["minimum"]
            if "maximum" in field_schema:
                prop["maximum"] = field_schema["maximum"]
            if "enum" in field_schema:
                prop["enum"] = field_schema["enum"]
            properties[field_name] = prop

            if field_schema.get("required", False):
                required.append(field_name)

        result = {"type": "object", "properties": properties}
        if required:
            result["required"] = required

        return result


def create_cyrp_api() -> APIServer:
    """创建穿黄工程API服务器"""
    server = APIServer()

    # 创建路由器
    # 认证路由
    auth_router = APIRouter(prefix="/api/v1/auth")

    @auth_router.post("/login", auth_required=False, summary="用户登录", tags=["认证"])
    async def login(request: APIRequest):
        username = request.body.get("username")
        password = request.body.get("password")

        # 模拟验证(实际应调用认证服务)
        if username == "admin" and password == "admin":
            token = server.jwt_manager.create_token(
                user_id="admin-001",
                username="admin",
                roles=["admin"]
            )
            return {"token": token, "expires_in": 86400}
        else:
            return APIError(
                code="INVALID_CREDENTIALS",
                message="用户名或密码错误",
                status_code=401
            ).to_response()

    @auth_router.post("/refresh", summary="刷新令牌", tags=["认证"])
    async def refresh_token(request: APIRequest):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            new_token = server.jwt_manager.refresh_token(token)
            if new_token:
                return {"token": new_token, "expires_in": 86400}
        return APIError(
            code="INVALID_TOKEN",
            message="令牌无效或已过期",
            status_code=401
        ).to_response()

    server.include_router(auth_router)

    # 数据路由
    data_router = APIRouter(prefix="/api/v1/data")

    @data_router.get("/realtime", summary="获取实时数据", tags=["数据"])
    async def get_realtime_data(request: APIRequest):
        return {
            "timestamp": datetime.now().isoformat(),
            "data": {
                "inlet_flow": 265.5,
                "outlet_flow": 264.8,
                "inlet_pressure": 0.35,
                "outlet_pressure": 0.30,
                "water_level": 8.5,
                "temperature": 18.2
            }
        }

    @data_router.get("/history", summary="获取历史数据", tags=["数据"])
    async def get_history_data(request: APIRequest):
        tag = request.query_params.get("tag", "inlet_flow")
        start = request.query_params.get("start")
        end = request.query_params.get("end")

        # 模拟历史数据
        return {
            "tag": tag,
            "start": start,
            "end": end,
            "data": [
                {"timestamp": "2024-01-01T00:00:00", "value": 265.0},
                {"timestamp": "2024-01-01T01:00:00", "value": 266.0},
                {"timestamp": "2024-01-01T02:00:00", "value": 264.5}
            ]
        }

    server.include_router(data_router)

    # 报警路由
    alarm_router = APIRouter(prefix="/api/v1/alarms")

    @alarm_router.get("", summary="获取报警列表", tags=["报警"])
    async def get_alarms(request: APIRequest):
        status = request.query_params.get("status", "active")
        return {
            "total": 2,
            "alarms": [
                {
                    "id": "ALM001",
                    "name": "高水位报警",
                    "severity": "high",
                    "status": "active",
                    "occurred_at": "2024-01-01T10:00:00"
                }
            ]
        }

    @alarm_router.post("/{alarm_id}/acknowledge", summary="确认报警", tags=["报警"])
    async def acknowledge_alarm(request: APIRequest):
        alarm_id = request.path_params.get("alarm_id")
        return {"message": f"报警 {alarm_id} 已确认"}

    server.include_router(alarm_router)

    # 设备路由
    equipment_router = APIRouter(prefix="/api/v1/equipment")

    @equipment_router.get("", summary="获取设备列表", tags=["设备"])
    async def get_equipment_list(request: APIRequest):
        return {
            "total": 10,
            "equipment": [
                {"id": "PUMP001", "name": "1#水泵", "status": "running"},
                {"id": "PUMP002", "name": "2#水泵", "status": "standby"},
                {"id": "VALVE001", "name": "进口阀门", "status": "open"}
            ]
        }

    @equipment_router.get("/{equipment_id}", summary="获取设备详情", tags=["设备"])
    async def get_equipment_detail(request: APIRequest):
        equipment_id = request.path_params.get("equipment_id")
        return {
            "id": equipment_id,
            "name": "1#水泵",
            "type": "pump",
            "status": "running",
            "parameters": {
                "speed": 1450,
                "power": 250,
                "vibration": 2.5,
                "temperature": 45
            }
        }

    @equipment_router.post("/{equipment_id}/control", summary="控制设备", tags=["设备"])
    async def control_equipment(request: APIRequest):
        equipment_id = request.path_params.get("equipment_id")
        action = request.body.get("action")
        return {"message": f"设备 {equipment_id} 执行操作: {action}"}

    server.include_router(equipment_router)

    # 报表路由
    report_router = APIRouter(prefix="/api/v1/reports")

    @report_router.get("", summary="获取报表列表", tags=["报表"])
    async def get_reports(request: APIRequest):
        return {
            "reports": [
                {"id": "RPT001", "name": "日报", "type": "daily"},
                {"id": "RPT002", "name": "周报", "type": "weekly"}
            ]
        }

    @report_router.post("/generate", summary="生成报表", tags=["报表"])
    async def generate_report(request: APIRequest):
        report_type = request.body.get("type", "daily")
        return {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "generating",
            "message": f"正在生成{report_type}报表"
        }

    server.include_router(report_router)

    # 系统路由
    system_router = APIRouter(prefix="/api/v1/system")

    @system_router.get("/health", auth_required=False, summary="健康检查", tags=["系统"])
    async def health_check(request: APIRequest):
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }

    @system_router.get("/info", summary="系统信息", tags=["系统"])
    async def system_info(request: APIRequest):
        return {
            "name": "穿黄工程智能管控系统",
            "version": "1.0.0",
            "uptime": "10d 5h 30m",
            "components": {
                "database": "connected",
                "communication": "connected",
                "alarm_system": "running"
            }
        }

    server.include_router(system_router)

    return server
