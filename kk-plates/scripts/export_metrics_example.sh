#!/bin/bash
# Example script to export metrics and send to Power BI

set -e

# Configuration
CONFIG_FILE="${CONFIG_FILE:-configs/default.yaml}"
DURATION="${DURATION:-10}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/metrics_export.json}"

echo "KK-Plates Metrics Export Example"
echo "================================"
echo "Config: $CONFIG_FILE"
echo "Duration: $DURATION seconds"
echo ""

# Check if the CLI is installed
if ! command -v kkplates &> /dev/null; then
    echo "Error: kkplates CLI not found. Please run 'make setup' first."
    exit 1
fi

# Export metrics
echo "Capturing metrics for $DURATION seconds..."
kkplates export-metrics \
    --config "$CONFIG_FILE" \
    --seconds "$DURATION" \
    --output "$OUTPUT_FILE"

# Check if export was successful
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "Metrics exported successfully to: $OUTPUT_FILE"
    echo ""
    echo "Sample metrics:"
    # Show first few lines of the output
    head -n 20 "$OUTPUT_FILE" | python -m json.tool
    
    # Example: Send to Power BI using curl
    # This is an example - replace with your actual Power BI endpoint
    echo ""
    echo "Example Power BI upload command:"
    echo "curl -X POST 'https://api.powerbi.com/v1.0/myorg/datasets/{dataset_id}/rows' \\"
    echo "  -H 'Authorization: Bearer {access_token}' \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d @$OUTPUT_FILE"
    
    # Example: Process metrics with jq
    if command -v jq &> /dev/null; then
        echo ""
        echo "Metrics summary:"
        jq '.stats' "$OUTPUT_FILE"
    fi
else
    echo "Error: Failed to export metrics"
    exit 1
fi

# Cleanup (optional)
# rm -f "$OUTPUT_FILE"

echo ""
echo "Done!"