#!/usr/bin/env bash
set -euo pipefail
lsblk
DEV="/dev/nvme1n1"   # your 200G EBS
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y xfsprogs
fi
sudo file -s "$DEV" || true
sudo mkfs -t xfs "$DEV" || true
sudo mkdir -p /mnt/data
UUID=$(sudo blkid -s UUID -o value "$DEV")
echo "UUID=$UUID /mnt/data xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
sudo mount -a
sudo mkdir -p /mnt/data/pdfs /mnt/data/reports
sudo chown -R $(id -un):$(id -gn) /mnt/data
echo "Mounted $DEV at /mnt/data"
