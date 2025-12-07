"""
Asset Management Module for CYRP
穿黄工程资产管理模块
"""

from cyrp.asset.asset_manager import (
    AssetType,
    AssetStatus,
    DepreciationMethod,
    MaintenanceType,
    SparePartStatus,
    Location,
    Manufacturer,
    AssetCategory,
    Asset,
    AssetTransaction,
    SparePart,
    MaintenanceRecord,
    AssetHealthScore,
    AssetRepository,
    SparePartManager,
    MaintenanceManager,
    AssetHealthAssessor,
    AssetReportGenerator,
    AssetManagementService,
    create_cyrp_asset_management,
)

__all__ = [
    "AssetType",
    "AssetStatus",
    "DepreciationMethod",
    "MaintenanceType",
    "SparePartStatus",
    "Location",
    "Manufacturer",
    "AssetCategory",
    "Asset",
    "AssetTransaction",
    "SparePart",
    "MaintenanceRecord",
    "AssetHealthScore",
    "AssetRepository",
    "SparePartManager",
    "MaintenanceManager",
    "AssetHealthAssessor",
    "AssetReportGenerator",
    "AssetManagementService",
    "create_cyrp_asset_management",
]
