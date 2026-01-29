"""Multi-tenant cloud service infrastructure for Clean.

This module provides the core infrastructure for running Clean as a
multi-tenant SaaS platform with workspaces, RBAC, and API management.

Example:
    >>> from clean.cloud import CloudService, Workspace, User
    >>>
    >>> service = CloudService(database_url="postgresql://...")
    >>> workspace = await service.create_workspace("my-org")
    >>> user = await service.create_user("user@example.com", workspace_id=workspace.id)
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class Role(Enum):
    """User roles for RBAC."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(Enum):
    """Granular permissions."""

    # Workspace management
    WORKSPACE_READ = "workspace:read"
    WORKSPACE_WRITE = "workspace:write"
    WORKSPACE_DELETE = "workspace:delete"
    WORKSPACE_SETTINGS = "workspace:settings"

    # User management
    USER_INVITE = "user:invite"
    USER_REMOVE = "user:remove"
    USER_ROLE = "user:role"

    # Data operations
    DATA_UPLOAD = "data:upload"
    DATA_READ = "data:read"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # Analysis
    ANALYSIS_RUN = "analysis:run"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_DELETE = "analysis:delete"

    # API keys
    API_KEY_CREATE = "api_key:create"
    API_KEY_REVOKE = "api_key:revoke"


# Role -> Permissions mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.OWNER: set(Permission),  # All permissions
    Role.ADMIN: {
        Permission.WORKSPACE_READ,
        Permission.WORKSPACE_WRITE,
        Permission.WORKSPACE_SETTINGS,
        Permission.USER_INVITE,
        Permission.USER_REMOVE,
        Permission.DATA_UPLOAD,
        Permission.DATA_READ,
        Permission.DATA_DELETE,
        Permission.DATA_EXPORT,
        Permission.ANALYSIS_RUN,
        Permission.ANALYSIS_READ,
        Permission.ANALYSIS_DELETE,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_REVOKE,
    },
    Role.MEMBER: {
        Permission.WORKSPACE_READ,
        Permission.DATA_UPLOAD,
        Permission.DATA_READ,
        Permission.DATA_EXPORT,
        Permission.ANALYSIS_RUN,
        Permission.ANALYSIS_READ,
    },
    Role.VIEWER: {
        Permission.WORKSPACE_READ,
        Permission.DATA_READ,
        Permission.ANALYSIS_READ,
    },
}


class SubscriptionTier(Enum):
    """Subscription tiers for billing."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Limits for each subscription tier."""

    max_workspaces: int
    max_users_per_workspace: int
    max_datasets: int
    max_dataset_size_mb: int
    max_analyses_per_month: int
    max_api_requests_per_day: int
    retention_days: int
    sso_enabled: bool
    audit_logs: bool
    priority_support: bool


TIER_LIMITS: dict[SubscriptionTier, TierLimits] = {
    SubscriptionTier.FREE: TierLimits(
        max_workspaces=1,
        max_users_per_workspace=3,
        max_datasets=10,
        max_dataset_size_mb=100,
        max_analyses_per_month=50,
        max_api_requests_per_day=100,
        retention_days=7,
        sso_enabled=False,
        audit_logs=False,
        priority_support=False,
    ),
    SubscriptionTier.STARTER: TierLimits(
        max_workspaces=3,
        max_users_per_workspace=10,
        max_datasets=50,
        max_dataset_size_mb=500,
        max_analyses_per_month=500,
        max_api_requests_per_day=1000,
        retention_days=30,
        sso_enabled=False,
        audit_logs=False,
        priority_support=False,
    ),
    SubscriptionTier.PROFESSIONAL: TierLimits(
        max_workspaces=10,
        max_users_per_workspace=50,
        max_datasets=200,
        max_dataset_size_mb=2000,
        max_analyses_per_month=5000,
        max_api_requests_per_day=10000,
        retention_days=90,
        sso_enabled=True,
        audit_logs=True,
        priority_support=False,
    ),
    SubscriptionTier.ENTERPRISE: TierLimits(
        max_workspaces=-1,  # Unlimited
        max_users_per_workspace=-1,
        max_datasets=-1,
        max_dataset_size_mb=-1,
        max_analyses_per_month=-1,
        max_api_requests_per_day=-1,
        retention_days=365,
        sso_enabled=True,
        audit_logs=True,
        priority_support=True,
    ),
}


@dataclass
class User:
    """User entity."""

    id: str
    email: str
    name: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    is_verified: bool = False
    password_hash: str | None = None
    sso_provider: str | None = None
    sso_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "sso_provider": self.sso_provider,
        }


@dataclass
class WorkspaceMember:
    """Workspace membership."""

    user_id: str
    workspace_id: str
    role: Role
    joined_at: datetime
    invited_by: str | None = None


@dataclass
class Workspace:
    """Workspace entity."""

    id: str
    name: str
    slug: str
    owner_id: str
    created_at: datetime
    updated_at: datetime
    tier: SubscriptionTier = SubscriptionTier.FREE
    is_active: bool = True
    settings: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tier": self.tier.value,
            "is_active": self.is_active,
            "settings": self.settings,
        }

    @property
    def limits(self) -> TierLimits:
        """Get tier limits."""
        return TIER_LIMITS[self.tier]


@dataclass
class APIKey:
    """API key for programmatic access."""

    id: str
    workspace_id: str
    name: str
    key_hash: str
    prefix: str  # First 8 chars for identification
    created_by: str
    created_at: datetime
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    is_active: bool = True
    scopes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "prefix": self.prefix,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
            "scopes": self.scopes,
        }


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance."""

    id: str
    workspace_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str | None
    timestamp: datetime
    ip_address: str | None = None
    user_agent: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "success"


@dataclass
class UsageRecord:
    """Usage tracking record for billing."""

    id: str
    workspace_id: str
    metric: str  # e.g., "analyses", "api_requests", "storage_mb"
    value: float
    timestamp: datetime
    period_start: datetime
    period_end: datetime


class StorageBackend(ABC):
    """Abstract storage backend for multi-tenant data."""

    @abstractmethod
    async def save_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
        data: bytes,
        metadata: dict[str, Any],
    ) -> str:
        """Save a dataset."""
        pass

    @abstractmethod
    async def load_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
    ) -> tuple[bytes, dict[str, Any]]:
        """Load a dataset."""
        pass

    @abstractmethod
    async def delete_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
    ) -> None:
        """Delete a dataset."""
        pass

    @abstractmethod
    async def list_datasets(
        self,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        """List datasets in a workspace."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend for development."""

    def __init__(self, base_path: str | None = None):
        """Initialize local storage.

        Args:
            base_path: Base path for storage. If None, uses system temp directory.
        """
        import os
        import tempfile
        import warnings

        if base_path is None:
            base_path = os.path.join(tempfile.gettempdir(), "clean-storage")
            warnings.warn(
                f"Using temporary storage at {base_path}. "
                "Data may be lost on reboot. Set base_path for persistent storage.",
                UserWarning,
                stacklevel=2,
            )

        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    async def save_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
        data: bytes,
        metadata: dict[str, Any],
    ) -> str:
        """Save dataset to local filesystem."""
        import json
        import os

        workspace_dir = os.path.join(self.base_path, workspace_id)
        os.makedirs(workspace_dir, exist_ok=True)

        data_path = os.path.join(workspace_dir, f"{dataset_id}.data")
        meta_path = os.path.join(workspace_dir, f"{dataset_id}.meta")

        with open(data_path, "wb") as f:
            f.write(data)

        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        return data_path

    async def load_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
    ) -> tuple[bytes, dict[str, Any]]:
        """Load dataset from local filesystem."""
        import json
        import os

        workspace_dir = os.path.join(self.base_path, workspace_id)
        data_path = os.path.join(workspace_dir, f"{dataset_id}.data")
        meta_path = os.path.join(workspace_dir, f"{dataset_id}.meta")

        with open(data_path, "rb") as f:
            data = f.read()

        with open(meta_path) as f:
            metadata = json.load(f)

        return data, metadata

    async def delete_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
    ) -> None:
        """Delete dataset from local filesystem."""
        import os

        workspace_dir = os.path.join(self.base_path, workspace_id)
        data_path = os.path.join(workspace_dir, f"{dataset_id}.data")
        meta_path = os.path.join(workspace_dir, f"{dataset_id}.meta")

        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

    async def list_datasets(
        self,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        """List datasets in workspace."""
        import json
        import os

        workspace_dir = os.path.join(self.base_path, workspace_id)
        if not os.path.exists(workspace_dir):
            return []

        datasets = []
        for filename in os.listdir(workspace_dir):
            if filename.endswith(".meta"):
                meta_path = os.path.join(workspace_dir, filename)
                with open(meta_path) as f:
                    metadata = json.load(f)
                datasets.append(metadata)

        return datasets


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend for production.

    Requires: pip install boto3
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "clean-data",
        region: str = "us-east-1",
        **kwargs: Any,
    ):
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all objects
            region: AWS region
            **kwargs: Additional boto3 client config
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.client_kwargs = kwargs
        self._client = None

    def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 storage. "
                    "Install with: pip install boto3"
                )
            self._client = boto3.client(
                "s3",
                region_name=self.region,
                **self.client_kwargs,
            )
        return self._client

    def _get_key(self, workspace_id: str, dataset_id: str, suffix: str) -> str:
        """Generate S3 key."""
        return f"{self.prefix}/{workspace_id}/{dataset_id}.{suffix}"

    async def save_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
        data: bytes,
        metadata: dict[str, Any],
    ) -> str:
        """Save dataset to S3."""
        import json

        client = self._get_client()
        data_key = self._get_key(workspace_id, dataset_id, "data")
        meta_key = self._get_key(workspace_id, dataset_id, "meta")

        client.put_object(Bucket=self.bucket, Key=data_key, Body=data)
        client.put_object(
            Bucket=self.bucket,
            Key=meta_key,
            Body=json.dumps(metadata).encode(),
        )

        return f"s3://{self.bucket}/{data_key}"

    async def load_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
    ) -> tuple[bytes, dict[str, Any]]:
        """Load dataset from S3."""
        import json

        client = self._get_client()
        data_key = self._get_key(workspace_id, dataset_id, "data")
        meta_key = self._get_key(workspace_id, dataset_id, "meta")

        data_resp = client.get_object(Bucket=self.bucket, Key=data_key)
        data = data_resp["Body"].read()

        meta_resp = client.get_object(Bucket=self.bucket, Key=meta_key)
        metadata = json.loads(meta_resp["Body"].read().decode())

        return data, metadata

    async def delete_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
    ) -> None:
        """Delete dataset from S3."""
        client = self._get_client()
        data_key = self._get_key(workspace_id, dataset_id, "data")
        meta_key = self._get_key(workspace_id, dataset_id, "meta")

        client.delete_objects(
            Bucket=self.bucket,
            Delete={"Objects": [{"Key": data_key}, {"Key": meta_key}]},
        )

    async def list_datasets(
        self,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        """List datasets in workspace."""
        import json

        client = self._get_client()
        prefix = f"{self.prefix}/{workspace_id}/"

        response = client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        datasets = []

        for obj in response.get("Contents", []):
            if obj["Key"].endswith(".meta"):
                meta_resp = client.get_object(Bucket=self.bucket, Key=obj["Key"])
                metadata = json.loads(meta_resp["Body"].read().decode())
                datasets.append(metadata)

        return datasets


class InMemoryDatabase:
    """In-memory database for testing and development."""

    def __init__(self):
        """Initialize in-memory storage."""
        self.users: dict[str, User] = {}
        self.workspaces: dict[str, Workspace] = {}
        self.memberships: list[WorkspaceMember] = []
        self.api_keys: dict[str, APIKey] = {}
        self.audit_logs: list[AuditLogEntry] = []
        self.usage_records: list[UsageRecord] = []


class CloudService:
    """Main cloud service for multi-tenant operations.

    Provides user management, workspace management, RBAC,
    and billing functionality.
    """

    def __init__(
        self,
        database_url: str | None = None,
        storage_backend: StorageBackend | None = None,
    ):
        """Initialize the cloud service.

        Args:
            database_url: Database connection URL (None for in-memory)
            storage_backend: Storage backend for datasets
        """
        self.database_url = database_url
        self.storage = storage_backend or LocalStorageBackend()

        # Use in-memory database for development
        self._db = InMemoryDatabase()

    # User Management

    async def create_user(
        self,
        email: str,
        name: str,
        password: str | None = None,
        sso_provider: str | None = None,
        sso_id: str | None = None,
    ) -> User:
        """Create a new user.

        Args:
            email: User email
            name: Display name
            password: Password (for email auth)
            sso_provider: SSO provider name
            sso_id: SSO user ID

        Returns:
            Created user
        """
        user_id = str(uuid.uuid4())
        now = datetime.utcnow()

        password_hash = None
        if password:
            password_hash = self._hash_password(password)

        user = User(
            id=user_id,
            email=email,
            name=name,
            created_at=now,
            updated_at=now,
            password_hash=password_hash,
            sso_provider=sso_provider,
            sso_id=sso_id,
        )

        self._db.users[user_id] = user
        return user

    async def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self._db.users.get(user_id)

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        for user in self._db.users.values():
            if user.email == email:
                return user
        return None

    async def authenticate_user(self, email: str, password: str) -> User | None:
        """Authenticate user with email/password."""
        user = await self.get_user_by_email(email)
        if not user or not user.password_hash:
            return None

        if self._verify_password(password, user.password_hash):
            return user
        return None

    # Workspace Management

    async def create_workspace(
        self,
        name: str,
        owner_id: str,
        tier: SubscriptionTier = SubscriptionTier.FREE,
    ) -> Workspace:
        """Create a new workspace.

        Args:
            name: Workspace name
            owner_id: Owner user ID
            tier: Subscription tier

        Returns:
            Created workspace
        """
        workspace_id = str(uuid.uuid4())
        slug = self._generate_slug(name)
        now = datetime.utcnow()

        workspace = Workspace(
            id=workspace_id,
            name=name,
            slug=slug,
            owner_id=owner_id,
            created_at=now,
            updated_at=now,
            tier=tier,
        )

        self._db.workspaces[workspace_id] = workspace

        # Add owner as member
        await self.add_workspace_member(workspace_id, owner_id, Role.OWNER)

        return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        """Get workspace by ID."""
        return self._db.workspaces.get(workspace_id)

    async def list_user_workspaces(self, user_id: str) -> list[Workspace]:
        """List workspaces a user belongs to."""
        workspace_ids = {
            m.workspace_id
            for m in self._db.memberships
            if m.user_id == user_id
        }
        return [
            w for w in self._db.workspaces.values()
            if w.id in workspace_ids
        ]

    async def add_workspace_member(
        self,
        workspace_id: str,
        user_id: str,
        role: Role,
        invited_by: str | None = None,
    ) -> WorkspaceMember:
        """Add a member to a workspace."""
        member = WorkspaceMember(
            user_id=user_id,
            workspace_id=workspace_id,
            role=role,
            joined_at=datetime.utcnow(),
            invited_by=invited_by,
        )
        self._db.memberships.append(member)
        return member

    async def get_member_role(
        self,
        workspace_id: str,
        user_id: str,
    ) -> Role | None:
        """Get a user's role in a workspace."""
        for member in self._db.memberships:
            if member.workspace_id == workspace_id and member.user_id == user_id:
                return member.role
        return None

    async def check_permission(
        self,
        workspace_id: str,
        user_id: str,
        permission: Permission,
    ) -> bool:
        """Check if user has a permission in workspace."""
        role = await self.get_member_role(workspace_id, user_id)
        if role is None:
            return False

        return permission in ROLE_PERMISSIONS[role]

    # API Key Management

    async def create_api_key(
        self,
        workspace_id: str,
        name: str,
        created_by: str,
        scopes: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> tuple[APIKey, str]:
        """Create a new API key.

        Args:
            workspace_id: Workspace ID
            name: Key name/description
            created_by: User ID who created the key
            scopes: Optional scope restrictions
            expires_in_days: Days until expiration

        Returns:
            Tuple of (APIKey, raw_key)
        """
        key_id = str(uuid.uuid4())
        raw_key = f"clnk_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_api_key(raw_key)
        prefix = raw_key[:12]

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            id=key_id,
            workspace_id=workspace_id,
            name=name,
            key_hash=key_hash,
            prefix=prefix,
            created_by=created_by,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            scopes=scopes or [],
        )

        self._db.api_keys[key_id] = api_key
        return api_key, raw_key

    async def validate_api_key(self, raw_key: str) -> APIKey | None:
        """Validate an API key.

        Args:
            raw_key: Raw API key string

        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = self._hash_api_key(raw_key)

        for api_key in self._db.api_keys.values():
            if api_key.key_hash == key_hash:
                if not api_key.is_active:
                    return None
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    return None

                # Update last used
                api_key.last_used_at = datetime.utcnow()
                return api_key

        return None

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._db.api_keys:
            self._db.api_keys[key_id].is_active = False
            return True
        return False

    # Audit Logging

    async def log_audit_event(
        self,
        workspace_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        status: str = "success",
    ) -> AuditLogEntry:
        """Log an audit event."""
        workspace = await self.get_workspace(workspace_id)
        if workspace and not workspace.limits.audit_logs:
            return  # Audit logs not enabled for this tier

        entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            timestamp=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            status=status,
        )

        self._db.audit_logs.append(entry)
        return entry

    async def get_audit_logs(
        self,
        workspace_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLogEntry]:
        """Get audit logs for a workspace."""
        logs = [
            log for log in self._db.audit_logs
            if log.workspace_id == workspace_id
        ]
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[offset : offset + limit]

    # Usage Tracking

    async def record_usage(
        self,
        workspace_id: str,
        metric: str,
        value: float,
    ) -> UsageRecord:
        """Record usage metric."""
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            period_end = period_start.replace(year=now.year + 1, month=1)
        else:
            period_end = period_start.replace(month=now.month + 1)

        record = UsageRecord(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            metric=metric,
            value=value,
            timestamp=now,
            period_start=period_start,
            period_end=period_end,
        )

        self._db.usage_records.append(record)
        return record

    async def get_usage_summary(
        self,
        workspace_id: str,
        period_start: datetime | None = None,
    ) -> dict[str, float]:
        """Get usage summary for a workspace."""
        now = datetime.utcnow()
        if period_start is None:
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        summary: dict[str, float] = {}

        for record in self._db.usage_records:
            if (
                record.workspace_id == workspace_id
                and record.timestamp >= period_start
            ):
                summary[record.metric] = summary.get(record.metric, 0) + record.value

        return summary

    async def check_usage_limit(
        self,
        workspace_id: str,
        metric: str,
    ) -> tuple[bool, float, float]:
        """Check if workspace is within usage limits.

        Returns:
            Tuple of (within_limit, current_usage, limit)
        """
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False, 0, 0

        limits = workspace.limits
        usage = await self.get_usage_summary(workspace_id)

        metric_map = {
            "analyses": limits.max_analyses_per_month,
            "api_requests": limits.max_api_requests_per_day,
            "datasets": limits.max_datasets,
        }

        limit = metric_map.get(metric, -1)
        current = usage.get(metric, 0)

        if limit == -1:  # Unlimited
            return True, current, -1

        return current < limit, current, limit

    # Helper Methods

    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        salt = secrets.token_hex(16)
        hash_val = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000,
        )
        return f"{salt}:{hash_val.hex()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against hash."""
        try:
            salt, hash_val = password_hash.split(":")
            computed = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode(),
                salt.encode(),
                100000,
            )
            return computed.hex() == hash_val
        except Exception:
            return False

    def _hash_api_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _generate_slug(self, name: str) -> str:
        """Generate URL-safe slug from name."""
        import re

        slug = name.lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        return f"{slug}-{secrets.token_hex(4)}"


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int = 60,
    ):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed.

        Args:
            key: Identifier (e.g., workspace_id or user_id)

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        if key in self._requests:
            self._requests[key] = [
                t for t in self._requests[key]
                if t > window_start
            ]
        else:
            self._requests[key] = []

        current_count = len(self._requests[key])

        if current_count >= self.max_requests:
            return False, 0

        return True, self.max_requests - current_count - 1

    def record(self, key: str) -> None:
        """Record a request.

        Args:
            key: Identifier
        """
        now = time.time()

        if key not in self._requests:
            self._requests[key] = []

        self._requests[key].append(now)
