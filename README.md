# iExam

After installing the necessary packages, you can run iExam via "python3 core.py".

# Install on Mac
- % sudo python3 -m pip install PyQt5
- % sudo python3 -m pip install opencv-python
- % sudo python3 -m pip install pytesseract
- % sudo python3 -m pip install torch
- % sudo python3 -m pip install torchvision
- % sudo python3 -m pip install matplotlib

- Install the tesseract binary via brew and make sure the command of "tesseract --help" can run.
  See https://brew.sh/, https://guides.library.illinois.edu/c.php?g=347520&p=4121425 and https://tesseract-ocr.github.io/tessdoc/Installation.html
- *Maybe* need to explicitly install the tesseract language package: % brew install tesseract-lang (installing tesseract-eng failed).

# Install on Windows (using terminal)
install virtualenv using pip: pip install virtualenv <br>
create loacl virtual environment: virtualenv env <br>
go to the path <b>/env/Scripts</b>, activate env using: <b>activate</b> <br>
- install relative python packages using pip: <br>
- % pip install PyQt5
- % pip install opencv-python
- % pip install pytesseract
- % pip install torch
- % pip install torchvision
- % pip install matplotlib
- Then you need to install tesseract from: https://tesseract-ocr.github.io/tessdoc/Installation.html ， https://github.com/UB-Mannheim/tesseract/wiki 

# Install on Linux (using terminal)
install virtualenv: sudo apt install python3-virtualenv <br>
create loacl virtual environment: virtualenv env <br>
go to the path <b>/env/bin</b>, activate env using: <b>source activate</b> <br>
- install relative python packages using pip: <br>
- % pip install PyQt5
- % pip install opencv-python
	（if get error use "pip uninstall opencv-python" "pip install opencv-contrib-python-headless"
	"sudo apt-get install libxcb-xinerama0"）
- % pip install pytesseract
- % pip install torch (or pip install torch --no-cache-dir)
- % pip install torchvision (or pip install torchvision --no-cache-dir)
- % pip install matplotlib

# How to clone and modify code

$ git clone https://YourUserName@github.com/VPRLab/ProjectName

// make you local change

$ git add .

$ git status //check the files you want to commit

$ git commit -a

$ git push origin main
