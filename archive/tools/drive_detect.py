#!/usr/bin/env python3
"""
Detect Google Drive mount path for a shared folder.
Auto-locates across common Drive for Desktop mount points.
"""

import sys
import os
import argparse
from pathlib import Path
import getpass


def detect_drive_path(folder_name: str, verbose: bool = True) -> Path:
    """
    Detect Google Drive mount path for a shared folder.
    
    Args:
        folder_name: Name of the shared folder (e.g., "Secondary Sources")
        verbose: Print search progress
        
    Returns:
        Path object to the resolved folder
        
    Raises:
        FileNotFoundError: If folder not found in any common location
    """
    username = getpass.getuser()
    
    # Common Drive for Desktop mount points
    search_paths = []
    
    # Windows paths
    if sys.platform == 'win32':
        search_paths.extend([
            # Drive for Desktop - Shared drives
            Path(f"G:/Shared drives/{folder_name}"),
            Path(f"H:/Shared drives/{folder_name}"),
            Path(f"I:/Shared drives/{folder_name}"),
            # Drive for Desktop - My Drive
            Path(f"G:/My Drive/{folder_name}"),
            Path(f"H:/My Drive/{folder_name}"),
            # AppData location (older Drive File Stream)
            Path(f"C:/Users/{username}/AppData/Local/Google/DriveFS"),
        ])
    
    # macOS paths
    elif sys.platform == 'darwin':
        search_paths.extend([
            # Drive for Desktop - Shared drives
            Path(f"/Volumes/GoogleDrive/Shared drives/{folder_name}"),
            Path(f"/Volumes/Google Drive/Shared drives/{folder_name}"),
            # CloudStorage location
            Path.home() / f"Library/CloudStorage/GoogleDrive-{username}@*/Shared drives/{folder_name}",
            Path.home() / f"Library/CloudStorage/GoogleDrive/Shared drives/{folder_name}",
        ])
    
    # Linux paths
    else:
        search_paths.extend([
            Path.home() / f"GoogleDrive/Shared drives/{folder_name}",
            Path.home() / f"Google Drive/Shared drives/{folder_name}",
        ])
    
    if verbose:
        print(f"Searching for Drive folder: '{folder_name}'")
        print(f"Checking {len(search_paths)} potential locations...\n")
    
    # Search for the folder
    for i, path in enumerate(search_paths, 1):
        # Expand glob patterns
        if '*' in str(path):
            parent = Path(str(path).split('*')[0])
            if parent.exists():
                pattern = str(path).split(str(parent))[1].lstrip('/')
                matches = list(parent.glob(pattern))
                if matches:
                    for match in matches:
                        if match.exists() and match.is_dir():
                            if verbose:
                                print(f"✅ Found: {match}")
                            return match
        else:
            if verbose:
                status = "✅ Found" if path.exists() else "❌ Not found"
                print(f"  [{i}/{len(search_paths)}] {status}: {path}")
            
            if path.exists() and path.is_dir():
                if verbose:
                    print(f"\n✅ SUCCESS: Resolved Drive path")
                    print(f"   {path}\n")
                return path
    
    # Not found
    raise FileNotFoundError(
        f"\n❌ ERROR: Could not find Drive folder '{folder_name}'\n\n"
        f"Searched {len(search_paths)} locations:\n" +
        "\n".join(f"  - {p}" for p in search_paths[:5]) +
        "\n\nTroubleshooting:\n"
        "  1. Ensure Google Drive for Desktop is running\n"
        "  2. Check that '{folder_name}' is shared with you\n"
        "  3. Verify the folder name is spelled correctly\n"
        "  4. Try accessing the folder in Drive web UI first\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Detect Google Drive mount path for a shared folder"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Name of the shared folder (e.g., 'Secondary Sources')"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress search progress output"
    )
    
    args = parser.parse_args()
    
    try:
        path = detect_drive_path(args.folder, verbose=not args.quiet)
        
        # Print final result
        print("=" * 70)
        print("DRIVE PATH DETECTED")
        print("=" * 70)
        print(f"Folder: {args.folder}")
        print(f"Path:   {path}")
        print(f"Exists: {path.exists()}")
        
        # Check if it's readable
        try:
            file_count = len(list(path.iterdir()))
            print(f"Files:  {file_count} items")
        except PermissionError:
            print("Files:  <permission denied>")
        except Exception as e:
            print(f"Files:  <error: {e}>")
        
        print("=" * 70)
        print(f"\n✅ Use this path in your commands:\n{path}\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
