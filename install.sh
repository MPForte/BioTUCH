#!/bin/bash

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# URL encoding function
urle () { 
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

echo ""
echo "======================================"
echo "  BioTUCH Installation Script"
echo "======================================"
echo ""

# Get credentials
echo -e "\n${BLUE}After registering at https://biotuch.is.tue.mpg.de/, provide your credentials:${NC}"
read -p "Username: " username
read -s -p "Password: " password
echo ""
username=$(urle "$username")
password=$(urle "$password")

# Download BioTUCH data
echo ""
print_info "Downloading BioTUCH data and models..."
wget --post-data "username=$username&password=$password" \
     'https://download.is.tue.mpg.de/download.php?domain=biotuch&resume=1&sfile=data.zip' \
     -O 'data.zip' --continue

if [ $? -ne 0 ]; then
    print_error "Download failed. Please check your credentials and try again."
    exit 1
fi

print_status "Download complete"

# Check for conda or mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    print_status "Found mamba, using it for faster installation"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    print_status "Found conda"
else
    print_error "Neither conda nor mamba found. Please install conda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo ""
print_info "Creating biotuch conda environment..."
$CONDA_CMD env create -f environment.yml
print_status "Conda environment created"

# Setup conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
echo ""
print_info "Activating biotuch environment..."
conda activate biotuch
print_status "Environment activated"

# Install networkx without dependencies
echo ""
print_info "Installing networkx correct version... ${RED}(Ignore pip error)${NC}"
pip install networkx==2.5
print_status "networkx installed"

# Extract data
echo ""
print_info "Extracting data and models..."
unzip -q data.zip
rm data.zip
print_status "Data extracted"

echo ""
echo "======================================"
echo "  Installation Complete!"
echo "======================================"
echo ""
print_status "BioTUCH environment is ready!"
echo ""
echo "To test the installation, run:"
echo -e " ${BLUE}conda activate biotuch${NC}"
echo -e " ${BLUE}python biotuch.py --cfg_file cfg_files/demo.yaml --output_folder data/demo/output_test${NC}"
echo ""
print_info "Check that data/demo/output_test matches data/demo/output"
echo ""
