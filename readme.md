## Super simple python3 example to run openvino through opencv

### pre-req

Install Anaconda and create a virtual environment.


Activate the virtual environment.


Install the requried pip packages

	pip install -r req.txt

Download the person-detection-0200 model

	omz_downloader --name person-detection-0200
 
 OR the command below - depending on where the omz_downloader where installed
 
 	~/.local/bin/omz_downloader --name person-detection-0200

Add your input and output rtmp urls into main.py

Start your program

	python3 main.py
