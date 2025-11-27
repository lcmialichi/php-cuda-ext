<?php

function cuda_get_device_count(): int
{
}
function cuda_get_device_info(int $deviceId = null): array
{
}
function cuda_set_device(int $deviceId): bool
{
}
function cuda_get_current_device(): int
{
}
function cuda_get_memory_info(): array
{
}
function cuda_device_reset(): array
{
}
function cuda_get_driver_version(): array
{
}
function cuda_get_runtime_version(): array
{
}
function cuda_synchronize(): bool
{
}
function cuda_get_last_error(): array
{
}
function cuda_clear_error(): array
{
}
function cuda_get_peer_access(int $device1, int $device2): array
{
}