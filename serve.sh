#!/bin/bash
echo Serving with context 250k
echo Preventing this Mac from going to sleep
caffeinate -i ./ds4-server --host 0.0.0.0 --warm-weights --ctx 250000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 81920
