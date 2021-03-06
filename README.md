# iExam

create a new directory using: mkdir dir_name <br>
cd dir_name <br>
git clone https://github.com/VPRLab/iExam.git <br>
install virtualenv: pip install virtualenv <br>
create loacl virtual environment: virtualenv env <br>
activate local environment <br>
cd iExam <br>
install all packages: pip install -r requirements.txt <br>
After installing the necessary packages, you can run iExam via: python3 core_os_version.py <br>
Details for different platforms shown below <br>

# Requirments:
- % student roster (first name + surname), file format is txt <br>
- % 5-min training video (manually clip a video segment before exam) <br>
- % online exam testing video <br>

# Install on Mac
install virtualenv: pip install virtualenv <br>
create loacl virtual environment: virtualenv env <br>
go to the path <b>/env/bin</b>, activate env using: <b>source activate</b> <br>
- % pip install PyQt5
- % pip install opencv-python
- % pip install pytesseract
- % pip install torch
- % pip install torchvision
- % pip install matplotlib

- Install the tesseract binary via brew and make sure the command of "tesseract --help" can run.
  See https://brew.sh/, https://guides.library.illinois.edu/c.php?g=347520&p=4121425 and https://tesseract-ocr.github.io/tessdoc/Installation.html
- *Maybe* need to explicitly install the tesseract language package: % brew install tesseract-lang (installing tesseract-eng failed).

# Install on Windows (using terminal)
install virtualenv using pip: pip install virtualenv <br>
create local virtual environment: $ virtualenv -p python3 iExamEnv <br>
$ source iExamEnv/bin/activate <br>
$ cd iExam <br>
$ pip install -r requirements.txt
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
