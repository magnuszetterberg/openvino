#!/bin/bash

echo "** Setting up packages **"
echo ""
sudo apt install python3-pip && pip install -r req.txt
echo ""
echo "*** Download networks for Object Detections ***"
~/.local/bin/omz_downloader --name person-detection-0200 && echo "" && echo "*** OpenVino Networks are downloaded ***"

