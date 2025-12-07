-- CYRP Database Initialization Script
-- 穿黄工程数据库初始化脚本

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================
-- 用户和权限表
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200),
    email VARCHAR(200),
    password_hash VARCHAR(500) NOT NULL,
    salt VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_locked BOOLEAN DEFAULT FALSE,
    locked_until TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    last_login TIMESTAMP,
    password_changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    role_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    permissions BIGINT NOT NULL,
    is_system BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role_id)
);

-- ============================================================
-- 报警表
-- ============================================================

CREATE TABLE IF NOT EXISTS alarm_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alarm_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    severity INTEGER NOT NULL,
    tag_name VARCHAR(100),
    high_limit FLOAT,
    low_limit FLOAT,
    high_high_limit FLOAT,
    low_low_limit FLOAT,
    deadband FLOAT DEFAULT 0,
    on_delay INTEGER DEFAULT 0,
    off_delay INTEGER DEFAULT 0,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alarm_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    instance_id VARCHAR(50) NOT NULL,
    alarm_id VARCHAR(50) NOT NULL,
    name VARCHAR(200),
    severity INTEGER NOT NULL,
    state INTEGER NOT NULL,
    value FLOAT,
    limit_value FLOAT,
    message TEXT,
    occurred_at TIMESTAMP NOT NULL,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100),
    cleared_at TIMESTAMP,
    shelved_until TIMESTAMP,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alarm_history_occurred_at ON alarm_history(occurred_at);
CREATE INDEX idx_alarm_history_alarm_id ON alarm_history(alarm_id);
CREATE INDEX idx_alarm_history_state ON alarm_history(state);

-- ============================================================
-- 审计日志表
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entry_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    user_id UUID,
    username VARCHAR(100),
    session_id VARCHAR(100),
    ip_address VARCHAR(50),
    resource VARCHAR(200),
    action VARCHAR(100),
    result VARCHAR(20),
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);

-- ============================================================
-- 配置表
-- ============================================================

CREATE TABLE IF NOT EXISTS configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(200) UNIQUE NOT NULL,
    value TEXT,
    value_type VARCHAR(20),
    scope VARCHAR(20),
    encrypted BOOLEAN DEFAULT FALSE,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS configuration_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(200) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    comment TEXT
);

-- ============================================================
-- 设备和维护表
-- ============================================================

CREATE TABLE IF NOT EXISTS equipment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    equipment_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    type VARCHAR(50),
    location VARCHAR(200),
    manufacturer VARCHAR(200),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    install_date DATE,
    specifications JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS maintenance_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(50) UNIQUE NOT NULL,
    equipment_id VARCHAR(50) REFERENCES equipment(equipment_id),
    task_type VARCHAR(50),
    priority INTEGER DEFAULT 3,
    description TEXT,
    scheduled_date DATE,
    completed_date DATE,
    assigned_to VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    estimated_duration_hours FLOAT,
    actual_duration_hours FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_maintenance_tasks_equipment_id ON maintenance_tasks(equipment_id);
CREATE INDEX idx_maintenance_tasks_status ON maintenance_tasks(status);
CREATE INDEX idx_maintenance_tasks_scheduled_date ON maintenance_tasks(scheduled_date);

-- ============================================================
-- 报表表
-- ============================================================

CREATE TABLE IF NOT EXISTS report_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    template_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    report_type VARCHAR(50),
    sections JSONB,
    styles JSONB,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS report_instances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    instance_id VARCHAR(50) UNIQUE NOT NULL,
    template_id VARCHAR(50) REFERENCES report_templates(template_id),
    title VARCHAR(300),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20),
    file_paths JSONB,
    parameters JSONB
);

-- ============================================================
-- 场景表
-- ============================================================

CREATE TABLE IF NOT EXISTS scenarios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50),
    description TEXT,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scenario_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id VARCHAR(50) UNIQUE NOT NULL,
    scenario_id VARCHAR(50) REFERENCES scenarios(scenario_id),
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    status VARCHAR(20),
    results JSONB,
    metrics JSONB
);

-- ============================================================
-- 初始数据
-- ============================================================

-- 插入默认角色
INSERT INTO roles (role_id, name, description, permissions, is_system) VALUES
('viewer', '查看员', '只能查看系统信息', 31, TRUE),
('operator', '操作员', '可以进行日常操作', 2047, TRUE),
('engineer', '工程师', '可以进行工程配置', 65535, TRUE),
('admin', '系统管理员', '拥有所有权限', 2147483647, TRUE)
ON CONFLICT (role_id) DO NOTHING;

-- 输出完成信息
DO $$
BEGIN
    RAISE NOTICE 'CYRP数据库初始化完成';
END $$;
