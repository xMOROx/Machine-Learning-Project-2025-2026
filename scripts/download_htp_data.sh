#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Download How-to-Probe datasets: ImageNet, COCO, VOC"
    echo ""
    echo "Options:"
    echo "  --imagenet    Download ImageNet dataset (requires manual setup)"
    echo "  --coco        Download COCO 2014 dataset"
    echo "  --voc         Download Pascal VOC dataset"
    echo "  --all         Download all datasets"
    echo "  --data-dir    Custom data directory (default: ${DATA_DIR})"
    echo "  -h, --help    Show this help message"
}

DOWNLOAD_IMAGENET=false
DOWNLOAD_COCO=false
DOWNLOAD_VOC=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --imagenet)
            DOWNLOAD_IMAGENET=true
            shift
            ;;
        --coco)
            DOWNLOAD_COCO=true
            shift
            ;;
        --voc)
            DOWNLOAD_VOC=true
            shift
            ;;
        --all)
            DOWNLOAD_IMAGENET=true
            DOWNLOAD_COCO=true
            DOWNLOAD_VOC=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if ! $DOWNLOAD_IMAGENET && ! $DOWNLOAD_COCO && ! $DOWNLOAD_VOC; then
    log_warning "No dataset specified. Use --help for usage information."
    exit 0
fi

ensure_dir "${DATA_DIR}"
ensure_dir "${DATA_DIR}/htp"

download_imagenet() {
    log_info "ImageNet dataset setup..."
    
    IMAGENET_DIR="${DATA_DIR}/htp/imagenet"
    
    if [ -d "${IMAGENET_DIR}/train" ] && [ -d "${IMAGENET_DIR}/val" ]; then
        log_warning "ImageNet already exists at ${IMAGENET_DIR}"
        return 0
    fi
    
    ensure_dir "${IMAGENET_DIR}"
    
    log_warning "ImageNet requires manual download due to license restrictions."
    log_info "1. Register at https://image-net.org/"
    log_info "2. Download ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar"
    log_info "3. Extract to ${IMAGENET_DIR}/train and ${IMAGENET_DIR}/val"
    log_info ""
    log_info "Expected structure:"
    log_info "  ${IMAGENET_DIR}/train/n01440764/*.JPEG"
    log_info "  ${IMAGENET_DIR}/val/n01440764/*.JPEG"
    
    return 0
}

download_coco() {
    log_info "Downloading COCO 2014 dataset..."
    
    COCO_DIR="${DATA_DIR}/htp/coco"
    
    if [ -d "${COCO_DIR}/train2014" ] && [ -d "${COCO_DIR}/val2014" ]; then
        log_warning "COCO 2014 already exists at ${COCO_DIR}"
        return 0
    fi
    
    ensure_dir "${COCO_DIR}"
    ensure_dir "${COCO_DIR}/annotations"
    
    log_info "Downloading COCO train2014..."
    wget -c https://images.cocodataset.org/zips/train2014.zip -P "${COCO_DIR}"
    
    log_info "Downloading COCO val2014..."
    wget -c https://images.cocodataset.org/zips/val2014.zip -P "${COCO_DIR}"
    
    log_info "Downloading COCO annotations..."
    wget -c https://images.cocodataset.org/annotations/annotations_trainval2014.zip -P "${COCO_DIR}"
    
    log_info "Extracting archives..."
    unzip -q "${COCO_DIR}/train2014.zip" -d "${COCO_DIR}"
    unzip -q "${COCO_DIR}/val2014.zip" -d "${COCO_DIR}"
    unzip -q "${COCO_DIR}/annotations_trainval2014.zip" -d "${COCO_DIR}"
    
    rm -f "${COCO_DIR}"/*.zip
    
    log_success "COCO 2014 downloaded successfully"
}

download_voc() {
    log_info "Downloading Pascal VOC dataset..."
    
    VOC_DIR="${DATA_DIR}/htp/voc"
    
    if [ -d "${VOC_DIR}/VOCdevkit" ]; then
        log_warning "Pascal VOC already exists at ${VOC_DIR}"
        return 0
    fi
    
    ensure_dir "${VOC_DIR}"
    
    log_info "Downloading VOC2007..."
    wget -c https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P "${VOC_DIR}"
    wget -c https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P "${VOC_DIR}"
    
    log_info "Downloading VOC2012..."
    wget -c https://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P "${VOC_DIR}"
    
    log_info "Extracting archives..."
    tar -xf "${VOC_DIR}/VOCtrainval_06-Nov-2007.tar" -C "${VOC_DIR}"
    tar -xf "${VOC_DIR}/VOCtest_06-Nov-2007.tar" -C "${VOC_DIR}"
    tar -xf "${VOC_DIR}/VOCtrainval_11-May-2012.tar" -C "${VOC_DIR}"
    
    rm -f "${VOC_DIR}"/*.tar
    
    log_success "Pascal VOC downloaded successfully"
}

log_info "Starting How-to-Probe data download..."
log_info "Data directory: ${DATA_DIR}"

if $DOWNLOAD_IMAGENET; then
    download_imagenet
fi

if $DOWNLOAD_COCO; then
    download_coco
fi

if $DOWNLOAD_VOC; then
    download_voc
fi

log_success "How-to-Probe data download completed"
