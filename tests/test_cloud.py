"""Tests for multi-tenant cloud service."""

import pytest
from datetime import datetime, timedelta

from clean.cloud import (
    APIKey,
    AuditLogEntry,
    CloudService,
    InMemoryDatabase,
    LocalStorageBackend,
    Permission,
    RateLimiter,
    Role,
    ROLE_PERMISSIONS,
    SubscriptionTier,
    TIER_LIMITS,
    User,
    Workspace,
    WorkspaceMember,
)


class TestRolePermissions:
    """Tests for RBAC permissions."""

    def test_owner_has_all_permissions(self):
        """Test owner has all permissions."""
        owner_perms = ROLE_PERMISSIONS[Role.OWNER]
        assert owner_perms == set(Permission)

    def test_viewer_has_limited_permissions(self):
        """Test viewer has read-only permissions."""
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.DATA_READ in viewer_perms
        assert Permission.DATA_UPLOAD not in viewer_perms
        assert Permission.ANALYSIS_RUN not in viewer_perms

    def test_member_can_upload_and_analyze(self):
        """Test member can upload data and run analysis."""
        member_perms = ROLE_PERMISSIONS[Role.MEMBER]
        assert Permission.DATA_UPLOAD in member_perms
        assert Permission.ANALYSIS_RUN in member_perms
        assert Permission.USER_INVITE not in member_perms


class TestTierLimits:
    """Tests for subscription tier limits."""

    def test_free_tier_limits(self):
        """Test free tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.FREE]
        assert limits.max_workspaces == 1
        assert limits.max_users_per_workspace == 3
        assert limits.sso_enabled is False

    def test_enterprise_unlimited(self):
        """Test enterprise tier has unlimited features."""
        limits = TIER_LIMITS[SubscriptionTier.ENTERPRISE]
        assert limits.max_workspaces == -1  # Unlimited
        assert limits.sso_enabled is True
        assert limits.audit_logs is True


class TestUser:
    """Tests for User entity."""

    def test_user_creation(self):
        """Test user creation."""
        user = User(
            id="user-1",
            email="test@example.com",
            name="Test User",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        assert user.is_active is True
        assert user.is_verified is False

    def test_user_to_dict(self):
        """Test user serialization."""
        user = User(
            id="user-1",
            email="test@example.com",
            name="Test User",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            password_hash="secret",  # Should not be in output
        )
        data = user.to_dict()
        assert "password_hash" not in data
        assert data["email"] == "test@example.com"


class TestWorkspace:
    """Tests for Workspace entity."""

    def test_workspace_creation(self):
        """Test workspace creation."""
        workspace = Workspace(
            id="ws-1",
            name="Test Org",
            slug="test-org-abc123",
            owner_id="user-1",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        assert workspace.tier == SubscriptionTier.FREE
        assert workspace.is_active is True

    def test_workspace_limits(self):
        """Test workspace limits property."""
        workspace = Workspace(
            id="ws-1",
            name="Test",
            slug="test",
            owner_id="user-1",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tier=SubscriptionTier.PROFESSIONAL,
        )
        assert workspace.limits.sso_enabled is True


class TestAPIKey:
    """Tests for APIKey entity."""

    def test_api_key_creation(self):
        """Test API key creation."""
        api_key = APIKey(
            id="key-1",
            workspace_id="ws-1",
            name="My API Key",
            key_hash="abc123",
            prefix="clnk_abc",
            created_by="user-1",
            created_at=datetime.utcnow(),
        )
        assert api_key.is_active is True
        assert api_key.last_used_at is None


class TestCloudService:
    """Tests for CloudService."""

    @pytest.fixture
    def service(self):
        """Create a cloud service instance."""
        return CloudService()

    @pytest.mark.asyncio
    async def test_create_user(self, service):
        """Test user creation."""
        user = await service.create_user(
            email="test@example.com",
            name="Test User",
            password="secret123",
        )
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.password_hash is not None

    @pytest.mark.asyncio
    async def test_get_user(self, service):
        """Test getting user by ID."""
        user = await service.create_user(
            email="test@example.com",
            name="Test User",
        )
        retrieved = await service.get_user(user.id)
        assert retrieved is not None
        assert retrieved.email == user.email

    @pytest.mark.asyncio
    async def test_get_user_by_email(self, service):
        """Test getting user by email."""
        await service.create_user(
            email="test@example.com",
            name="Test User",
        )
        user = await service.get_user_by_email("test@example.com")
        assert user is not None

    @pytest.mark.asyncio
    async def test_authenticate_user(self, service):
        """Test user authentication."""
        await service.create_user(
            email="test@example.com",
            name="Test User",
            password="secret123",
        )

        # Valid credentials
        user = await service.authenticate_user("test@example.com", "secret123")
        assert user is not None

        # Invalid credentials
        user = await service.authenticate_user("test@example.com", "wrong")
        assert user is None

    @pytest.mark.asyncio
    async def test_create_workspace(self, service):
        """Test workspace creation."""
        user = await service.create_user(
            email="owner@example.com",
            name="Owner",
        )
        workspace = await service.create_workspace(
            name="My Workspace",
            owner_id=user.id,
        )

        assert workspace.id is not None
        assert workspace.owner_id == user.id
        assert "my-workspace" in workspace.slug.lower()

    @pytest.mark.asyncio
    async def test_workspace_membership(self, service):
        """Test workspace membership."""
        user = await service.create_user(
            email="owner@example.com",
            name="Owner",
        )
        workspace = await service.create_workspace(
            name="Test Workspace",
            owner_id=user.id,
        )

        # Owner should have OWNER role
        role = await service.get_member_role(workspace.id, user.id)
        assert role == Role.OWNER

    @pytest.mark.asyncio
    async def test_add_workspace_member(self, service):
        """Test adding workspace member."""
        owner = await service.create_user(
            email="owner@example.com",
            name="Owner",
        )
        member = await service.create_user(
            email="member@example.com",
            name="Member",
        )
        workspace = await service.create_workspace(
            name="Test Workspace",
            owner_id=owner.id,
        )

        await service.add_workspace_member(
            workspace.id,
            member.id,
            Role.MEMBER,
            invited_by=owner.id,
        )

        role = await service.get_member_role(workspace.id, member.id)
        assert role == Role.MEMBER

    @pytest.mark.asyncio
    async def test_check_permission(self, service):
        """Test permission checking."""
        owner = await service.create_user(
            email="owner@example.com",
            name="Owner",
        )
        viewer = await service.create_user(
            email="viewer@example.com",
            name="Viewer",
        )
        workspace = await service.create_workspace(
            name="Test Workspace",
            owner_id=owner.id,
        )

        await service.add_workspace_member(
            workspace.id,
            viewer.id,
            Role.VIEWER,
        )

        # Owner can upload
        assert await service.check_permission(
            workspace.id, owner.id, Permission.DATA_UPLOAD
        )

        # Viewer cannot upload
        assert not await service.check_permission(
            workspace.id, viewer.id, Permission.DATA_UPLOAD
        )

        # Viewer can read
        assert await service.check_permission(
            workspace.id, viewer.id, Permission.DATA_READ
        )

    @pytest.mark.asyncio
    async def test_list_user_workspaces(self, service):
        """Test listing user workspaces."""
        user = await service.create_user(
            email="user@example.com",
            name="User",
        )
        await service.create_workspace("Workspace 1", user.id)
        await service.create_workspace("Workspace 2", user.id)

        workspaces = await service.list_user_workspaces(user.id)
        assert len(workspaces) == 2


class TestAPIKeyManagement:
    """Tests for API key management."""

    @pytest.fixture
    def service(self):
        """Create a cloud service instance."""
        return CloudService()

    @pytest.mark.asyncio
    async def test_create_api_key(self, service):
        """Test API key creation."""
        user = await service.create_user(
            email="user@example.com",
            name="User",
        )
        workspace = await service.create_workspace("Test", user.id)

        api_key, raw_key = await service.create_api_key(
            workspace_id=workspace.id,
            name="Test Key",
            created_by=user.id,
        )

        assert api_key.id is not None
        assert raw_key.startswith("clnk_")
        assert api_key.prefix == raw_key[:12]

    @pytest.mark.asyncio
    async def test_validate_api_key(self, service):
        """Test API key validation."""
        user = await service.create_user(
            email="user@example.com",
            name="User",
        )
        workspace = await service.create_workspace("Test", user.id)

        api_key, raw_key = await service.create_api_key(
            workspace_id=workspace.id,
            name="Test Key",
            created_by=user.id,
        )

        # Valid key
        validated = await service.validate_api_key(raw_key)
        assert validated is not None
        assert validated.id == api_key.id

        # Invalid key
        validated = await service.validate_api_key("invalid_key")
        assert validated is None

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, service):
        """Test API key revocation."""
        user = await service.create_user(
            email="user@example.com",
            name="User",
        )
        workspace = await service.create_workspace("Test", user.id)

        api_key, raw_key = await service.create_api_key(
            workspace_id=workspace.id,
            name="Test Key",
            created_by=user.id,
        )

        # Revoke
        result = await service.revoke_api_key(api_key.id)
        assert result is True

        # Validate should fail
        validated = await service.validate_api_key(raw_key)
        assert validated is None

    @pytest.mark.asyncio
    async def test_api_key_expiration(self, service):
        """Test API key expiration."""
        user = await service.create_user(
            email="user@example.com",
            name="User",
        )
        workspace = await service.create_workspace("Test", user.id)

        api_key, raw_key = await service.create_api_key(
            workspace_id=workspace.id,
            name="Test Key",
            created_by=user.id,
            expires_in_days=30,
        )

        assert api_key.expires_at is not None
        assert api_key.expires_at > datetime.utcnow()


class TestAuditLogging:
    """Tests for audit logging."""

    @pytest.fixture
    def service(self):
        """Create a cloud service instance."""
        return CloudService()

    @pytest.mark.asyncio
    async def test_log_audit_event(self, service):
        """Test logging audit event."""
        user = await service.create_user(
            email="user@example.com",
            name="User",
        )
        workspace = await service.create_workspace(
            "Test",
            user.id,
            tier=SubscriptionTier.PROFESSIONAL,  # Has audit logs
        )

        await service.log_audit_event(
            workspace_id=workspace.id,
            user_id=user.id,
            action="data.upload",
            resource_type="dataset",
            resource_id="ds-1",
        )

        logs = await service.get_audit_logs(workspace.id)
        assert len(logs) == 1
        assert logs[0].action == "data.upload"


class TestUsageTracking:
    """Tests for usage tracking."""

    @pytest.fixture
    def service(self):
        """Create a cloud service instance."""
        return CloudService()

    @pytest.mark.asyncio
    async def test_record_usage(self, service):
        """Test recording usage."""
        user = await service.create_user(
            email="user@example.com",
            name="User",
        )
        workspace = await service.create_workspace("Test", user.id)

        await service.record_usage(workspace.id, "analyses", 1)
        await service.record_usage(workspace.id, "analyses", 1)

        summary = await service.get_usage_summary(workspace.id)
        assert summary["analyses"] == 2

    @pytest.mark.asyncio
    async def test_check_usage_limit(self, service):
        """Test usage limit checking."""
        user = await service.create_user(
            email="user@example.com",
            name="User",
        )
        workspace = await service.create_workspace("Test", user.id)

        # Free tier: 50 analyses per month
        for _ in range(45):
            await service.record_usage(workspace.id, "analyses", 1)

        within_limit, current, limit = await service.check_usage_limit(
            workspace.id, "analyses"
        )
        assert within_limit is True
        assert current == 45
        assert limit == 50


class TestRateLimiter:
    """Tests for rate limiter."""

    def test_allow_requests_within_limit(self):
        """Test allowing requests within limit."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        for _ in range(10):
            allowed, _ = limiter.check("user-1")
            assert allowed
            limiter.record("user-1")

    def test_block_requests_over_limit(self):
        """Test blocking requests over limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            limiter.record("user-1")

        allowed, remaining = limiter.check("user-1")
        assert allowed is False
        assert remaining == 0

    def test_separate_keys(self):
        """Test separate rate limits per key."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            limiter.record("user-1")

        # user-1 is limited
        allowed, _ = limiter.check("user-1")
        assert allowed is False

        # user-2 is not limited
        allowed, _ = limiter.check("user-2")
        assert allowed is True


class TestLocalStorageBackend:
    """Tests for local storage backend."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a storage backend."""
        return LocalStorageBackend(str(tmp_path / "storage"))

    @pytest.mark.asyncio
    async def test_save_and_load_dataset(self, storage):
        """Test saving and loading dataset."""
        data = b"test data content"
        metadata = {"name": "test.csv", "size": len(data)}

        path = await storage.save_dataset(
            workspace_id="ws-1",
            dataset_id="ds-1",
            data=data,
            metadata=metadata,
        )

        loaded_data, loaded_metadata = await storage.load_dataset(
            workspace_id="ws-1",
            dataset_id="ds-1",
        )

        assert loaded_data == data
        assert loaded_metadata["name"] == "test.csv"

    @pytest.mark.asyncio
    async def test_delete_dataset(self, storage):
        """Test deleting dataset."""
        await storage.save_dataset(
            workspace_id="ws-1",
            dataset_id="ds-1",
            data=b"data",
            metadata={},
        )

        await storage.delete_dataset("ws-1", "ds-1")

        with pytest.raises(FileNotFoundError):
            await storage.load_dataset("ws-1", "ds-1")

    @pytest.mark.asyncio
    async def test_list_datasets(self, storage):
        """Test listing datasets."""
        for i in range(3):
            await storage.save_dataset(
                workspace_id="ws-1",
                dataset_id=f"ds-{i}",
                data=b"data",
                metadata={"id": f"ds-{i}"},
            )

        datasets = await storage.list_datasets("ws-1")
        assert len(datasets) == 3
