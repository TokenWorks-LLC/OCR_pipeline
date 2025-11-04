#!/usr/bin/env python3
"""Test how BlockRole enum converts to string."""

from enum import Enum

class BlockRole(str, Enum):
    REFERENCE_META = 'reference_meta'

role = BlockRole.REFERENCE_META

print(f'str(role) = {str(role)}')
print(f'role.value = {role.value}')
print(f'Compare: str(role) in ["reference_meta"] = {str(role) in ["reference_meta"]}')
print(f'Compare: role.value in ["reference_meta"] = {role.value in ["reference_meta"]}')
